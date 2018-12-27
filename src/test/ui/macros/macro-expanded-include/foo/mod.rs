// ignore-test

macro_rules! m {
    () => { include!("file.txt"); }
}

macro_rules! n {
    () => { unsafe { asm!(include_str!("file.txt")); } }
}
