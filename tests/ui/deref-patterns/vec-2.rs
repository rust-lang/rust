// edition: 2021
// check-pass
// FIXME(deref_patterns): we aren't generating the false edge here,
// which means that borrowck doesn't run on all arms, as shown here
// with using an uninitialized local.
#![feature(forced_keywords)]

fn foo(x: i32) { println!("{x}") }

fn main() {
    let x: i32;
    match vec![ 1 ] {
        k#deref _ => (),
        _ => panic!("{x}"),
    }
}
