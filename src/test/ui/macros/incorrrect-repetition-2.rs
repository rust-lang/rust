macro_rules! foo {
    ($($a:expr)*) => {};
    //~^ WARN `$a:expr` is followed (through repetition) by itself, which is not allowed for
}
