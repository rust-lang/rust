// edition: 2021
// check-pass
#![feature(forced_keywords)]

fn foo(x: i32) { println!("{x}") }

fn main() {
    match vec![ vec![ 1 ] ] {
        k#deref [ _ ] => (),
        k#deref [ k#deref [ x ] ] => panic!("{x}"),
        _ => panic!(),
    }
}
