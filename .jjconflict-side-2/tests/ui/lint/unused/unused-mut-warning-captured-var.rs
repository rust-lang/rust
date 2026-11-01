//@ run-rustfix

#![forbid(unused_mut)]

fn main() {
    let mut x = 1;
    //~^ ERROR: variable does not need to be mutable
    (move|| { println!("{}", x); })();
}
