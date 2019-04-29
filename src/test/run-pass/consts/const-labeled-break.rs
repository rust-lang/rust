// Using labeled break in a while loop has caused an illegal instruction being
// generated, and an ICE later.
//
// See https://github.com/rust-lang/rust/issues/51350 for more information.

const CRASH: () = 'a: while break 'a {};

fn main() {
    println!("{:?}", CRASH);
}
