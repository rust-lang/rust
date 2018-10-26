macro_rules! foo {
    ($($a:expr)*) => {};
    //~^ ERROR `$a:expr` is followed (through repetition) by itself, which is not allowed for
}
