## Memory ownership

When a stack frame is exited, its local allocations are all released, and its
references to boxes are dropped.
