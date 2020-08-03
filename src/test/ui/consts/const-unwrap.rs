// check-fail

#![feature(const_option)]

const FOO: i32 = Some(42i32).unwrap();

// This causes an error, but it is attributed to the `panic` *inside* `Option::unwrap` (maybe due
// to `track_caller`?). A note points to the originating `const`.
const BAR: i32 = Option::<i32>::None.unwrap(); //~ NOTE

fn main() {
    println!("{}", FOO);
    println!("{}", BAR);
}
