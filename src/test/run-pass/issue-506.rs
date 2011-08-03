/*
  A reduced test case for Issue #506, provided by Rob Arnold.
*/

native "rust" mod rustrt {
    fn task_yield();
}

fn main() { spawn rustrt::task_yield(); }

