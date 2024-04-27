//@ edition:2021

#![allow(unused_macros)]
macro_rules! foo { ($x:pat | $y:pat) => {} } //~ ERROR `$x:pat` is followed by `|`, which is not allowed for `pat` fragments
macro_rules! baz { ($x:pat_param | $y:pat_param) => {} } // should be ok
macro_rules! qux { ($x:pat_param | $y:pat) => {} } // should be ok
macro_rules! ogg { ($x:pat | $y:pat_param) => {} } //~ ERROR `$x:pat` is followed by `|`, which is not allowed for `pat` fragments
macro_rules! match_any {
    ( $expr:expr , $( $( $pat:pat)|+ => $expr_arm:pat),+ ) => { //~ ERROR  `$pat:pat` may be followed by `|`, which is not allowed for `pat` fragments
        match $expr {
            $(
                $( $pat => $expr_arm, )+
            )+
        }
    };
}

fn main() {
    let result: Result<i64, i32> = Err(42);
    let int: i64 = match_any!(result, Ok(i) | Err(i) => i.into());
    assert_eq!(int, 42);
}
