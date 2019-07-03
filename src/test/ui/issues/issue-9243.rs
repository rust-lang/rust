// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Regression test for issue 9243
#![allow(non_upper_case_globals)]

pub struct Test {
    mem: isize,
}

pub static g_test: Test = Test {mem: 0};

impl Drop for Test {
    fn drop(&mut self) {}
}

fn main() {}
