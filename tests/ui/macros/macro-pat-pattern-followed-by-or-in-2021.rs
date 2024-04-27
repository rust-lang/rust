//@ edition:2021
#![allow(unused_macros)]
macro_rules! foo { ($x:pat | $y:pat) => {} } //~ ERROR `$x:pat` is followed by `|`, which is not allowed for `pat` fragments
macro_rules! bar { ($($x:pat)+ | $($y:pat)+) => {} } //~ ERROR `$x:pat` is followed by `|`, which is not allowed for `pat` fragments
macro_rules! qux { ($x:pat, $y:pat) => {} } // should be ok
macro_rules! match_any {
    ( $expr:expr , $( $( $pat:pat )|+ => $expr_arm:expr ),+ ) => { //~ ERROR `$pat:pat` may be followed by `|`, which is not allowed for `pat` fragments
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
