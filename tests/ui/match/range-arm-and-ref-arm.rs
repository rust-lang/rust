//! Regression test for <https://github.com/rust-lang/rust/issues/35423>.
//! This used to ICE.
//@ run-pass

fn main () {
    let x = 4;
    match x {
        ref r if *r < 0 => println!("got negative num {} < 0", r),
        e @ 1 ..= 100 => println!("got number within range [1,100] {}", e),
        _ => println!("no"),
    }
}
