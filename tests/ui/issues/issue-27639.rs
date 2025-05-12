//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

fn main() {
    const iter: i32 = 0;

    for i in 1..10 {
        println!("{}", i);
    }
}
