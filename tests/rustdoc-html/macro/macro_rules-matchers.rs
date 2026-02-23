// This is a regression test for issue #86208.
// It is also a general test of macro_rules! display.

#![crate_name = "foo"]

//@ has 'foo/macro.todo.html' '//pre' 'macro_rules! todo { \
//      () => { ... }; \
//      ($($arg:tt)+) => { ... }; \
// }'
pub use std::todo;

mod mod1 {
    //@ has 'foo/macro.macro1.html' '//pre' 'macro_rules! macro1 { \
    //      () => { ... }; \
    //      ($($arg:expr),+) => { ... }; \
    // }'
    #[macro_export]
    macro_rules! macro1 {
        () => {};
        ($($arg:expr),+) => { stringify!($($arg),+) };
    }
}
