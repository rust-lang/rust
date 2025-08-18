//@ run-pass
//@ check-run-results
#![feature(macro_attr)]
#![warn(unused)]

#[macro_export]
macro_rules! exported_attr {
    attr($($args:tt)*) { $($body:tt)* } => {
        println!(
            "exported_attr: args={:?}, body={:?}",
            stringify!($($args)*),
            stringify!($($body)*),
        );
    };
    { $($args:tt)* } => {
        println!("exported_attr!({:?})", stringify!($($args)*));
    };
    attr() {} => {
        unused_rule();
    };
    attr() {} => {
        compile_error!();
    };
    {} => {
        unused_rule();
    };
    {} => {
        compile_error!();
    };
}

macro_rules! local_attr {
    attr($($args:tt)*) { $($body:tt)* } => {
        println!(
            "local_attr: args={:?}, body={:?}",
            stringify!($($args)*),
            stringify!($($body)*),
        );
    };
    { $($args:tt)* } => {
        println!("local_attr!({:?})", stringify!($($args)*));
    };
    attr() {} => { //~ WARN: never used
        unused_rule();
    };
    attr() {} => {
        compile_error!();
    };
    {} => { //~ WARN: never used
        unused_rule();
    };
    {} => {
        compile_error!();
    };
}

fn main() {
    #[crate::exported_attr]
    struct S;
    #[::exported_attr(arguments, key = "value")]
    fn func(_arg: u32) {}
    #[self::exported_attr(1)]
    #[self::exported_attr(2)]
    struct Twice;

    crate::exported_attr!();
    crate::exported_attr!(invoked, arguments);

    #[exported_attr]
    struct S;
    #[exported_attr(arguments, key = "value")]
    fn func(_arg: u32) {}
    #[exported_attr(1)]
    #[exported_attr(2)]
    struct Twice;

    exported_attr!();
    exported_attr!(invoked, arguments);

    #[local_attr]
    struct S;
    #[local_attr(arguments, key = "value")]
    fn func(_arg: u32) {}
    #[local_attr(1)]
    #[local_attr(2)]
    struct Twice;

    local_attr!();
    local_attr!(invoked, arguments);
}
