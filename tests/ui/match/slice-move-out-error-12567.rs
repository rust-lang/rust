//! Regression test for https://github.com/rust-lang/rust/issues/12567

fn match_vecs<'a, T>(l1: &'a [T], l2: &'a [T]) {
    match (l1, l2) {
    //~^ ERROR: cannot move out of type `[T]`, a non-copy slice
    //~| ERROR: cannot move out of type `[T]`, a non-copy slice
        (&[], &[]) => println!("both empty"),
        (&[], &[hd, ..]) | (&[hd, ..], &[])
            => println!("one empty"),
        (&[hd1, ..], &[hd2, ..])
            => println!("both nonempty"),
    }
}

fn main() {}
