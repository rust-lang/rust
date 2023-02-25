// check-pass
#[macro_export(hello, world)] //~ WARN `#[macro_export]` can only take 1 or 0 arguments
macro_rules! a {
    () => ()
}

#[macro_export(not_local_inner_macros)] //~ WARN `not_local_inner_macros` isn't a valid `#[macro_export]` argument
macro_rules! b {
    () => ()
}

#[macro_export]
macro_rules! c {
    () => ()
}
#[macro_export(local_inner_macros)]
macro_rules! d {
    () => ()
}

#[macro_export()]
macro_rules! e {
    () => ()
}

fn main() {}
