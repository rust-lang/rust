// Regression test for issue 9243

pub struct Test {
    mem: isize,
}

pub static g_test: Test = Test {mem: 0};

impl Drop for Test {
    fn drop(&mut self) {}
}

fn main() {}
