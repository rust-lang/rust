This directory contains the robustness test for prallel front end, which means deadlocks
and other ice bugs. In other words, we don't care whether the compiler output in these tests,
but whether they can compile normally without deadlock or other ice bugs.

So when a test in this directory fails, please pay attention to whether it causes any ice problems.
If so(it should do), please post your comments in the issue corresponding to each test (or create a new issue
with the `wg-parallel-rustc` label). Even if it is an existing issue, please add a new comment,
which will help us determine the reproducibility of the bug.
