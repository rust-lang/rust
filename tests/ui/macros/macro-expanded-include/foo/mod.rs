//@ ignore-test (auxiliary, used by other tests)

macro_rules! m {
    () => { include!("file.txt"); }
}

macro_rules! n {
    () => { unsafe { core::arch::asm!(include_str!("file.txt")); } }
}
