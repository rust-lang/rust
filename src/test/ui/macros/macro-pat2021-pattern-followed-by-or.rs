#![feature(edition_macro_pats)]
#![allow(unused_macros)]
macro_rules! foo { ($x:pat2021 | $y:pat2021) => {} } //~ ERROR `$x:pat2021` is followed by `|`, which is not allowed for `pat2021` fragments
macro_rules! baz { ($x:pat2015 | $y:pat2015) => {} } // should be ok
macro_rules! qux { ($x:pat2015 | $y:pat2021) => {} } // should be ok
macro_rules! ogg { ($x:pat2021 | $y:pat2015) => {} } //~ ERROR `$x:pat2021` is followed by `|`, which is not allowed for `pat2021` fragments
macro_rules! match_any {
    ( $expr:expr , $( $( $pat:pat2021 )|+ => $expr_arm:pat2021 ),+ ) => { //~ ERROR  `$pat:pat2021` may be followed by `|`, which is not allowed for `pat2021` fragments
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
