// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

// Test that `Fn(isize) -> isize + 'static` parses as `(Fn(isize) -> isize) +
// 'static` and not `Fn(isize) -> (isize + 'static)`. The latter would
// cause a compilation error. Issue #18772.

fn adder(y: isize) -> Box<dyn Fn(isize) -> isize + 'static> {
    Box::new(move |x| y + x)
}

fn main() {}
