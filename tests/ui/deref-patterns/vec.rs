// edition: 2021
// check-pass
#![feature(forced_keywords)]

fn foo(x: i32) { println!("{x}") }

fn main() {
    match vec![ vec![], vec![1, 2, 3] ] {
        k#deref [ k#deref [], k#deref [1, x, y] ] => {
            foo(y - x)
        }
        k#deref [ ] => panic!(),
        k#deref [ ref x ] => panic!("{x:?}"),
        k#deref [ k#deref [ x ] ] => panic!("{x}"),
        _ => panic!(),
    }
}
