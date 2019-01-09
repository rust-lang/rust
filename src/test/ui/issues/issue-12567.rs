#![feature(slice_patterns)]

fn match_vecs<'a, T>(l1: &'a [T], l2: &'a [T]) {
    match (l1, l2) {
        (&[], &[]) => println!("both empty"),
        (&[], &[hd, ..]) | (&[hd, ..], &[])
            => println!("one empty"),
        //~^^ ERROR: cannot move out of type `[T]`, a non-copy slice
        //~^^^ ERROR: cannot move out of type `[T]`, a non-copy slice
        (&[hd1, ..], &[hd2, ..])
            => println!("both nonempty"),
        //~^^ ERROR: cannot move out of type `[T]`, a non-copy slice
        //~^^^ ERROR: cannot move out of type `[T]`, a non-copy slice
    }
}

fn main() {}
