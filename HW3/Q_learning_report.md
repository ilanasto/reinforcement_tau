# Q-Learning Report

## Success Rate
- Tabular Q-learning reached a success rate of 6.65% over 2000 episodes.
- The neural-network Q-learning implementation reached a success rate of 2.15% over the same 2000 episodes.
- We increased the number of episodes because the initial run produced an all-zero Q-table and did not learn effectively.
- This is a reasonable choice for FrozenLake, since the environment is stochastic and rewards are sparse. More episodes give the agent more opportunities to explore and update its Q-values, so the learning process becomes meaningful instead of remaining stuck at zero.
- In this run, the tabular method outperformed the simple one-layer neural-network version, but both implementations follow the same Q-learning idea.

## Final Q-Table
```text
[[9.77902274e-02 9.62734680e-02 3.40971475e-01 9.79730893e-02]
 [9.58607973e-03 9.76403428e-04 6.01365023e-03 1.22330059e-01]
 [1.56502174e-02 1.05934718e-02 6.39803047e-03 3.94400608e-02]
 [1.68278119e-02 8.32924766e-03 2.25936038e-02 2.65525431e-02]
 [4.57318981e-01 3.41819091e-02 3.58075294e-02 9.65713136e-04]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [4.23950315e-02 7.15974045e-04 5.80605120e-04 3.85334248e-05]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [9.89173400e-02 5.75245828e-02 3.85615230e-03 6.72290944e-01]
 [5.52412123e-02 8.24766373e-01 1.01267383e-01 9.44764433e-04]
 [9.32512352e-01 1.07778562e-03 1.66832884e-04 8.86931054e-04]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.21369888e-01 1.02273816e-01 6.94294873e-01 7.41041698e-02]
 [1.13643801e-01 9.82680875e-01 1.64151188e-01 1.43587064e-01]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
```
