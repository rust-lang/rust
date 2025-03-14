//@ build-fail
#![recursion_limit = "7"]

struct Thing<T>(T);

impl<T> Thing<T> {
    const X: usize = Thing::<Option<T>>::X;
}

fn main() {
    println!("{}", Thing::<i32>::X); //~ ERROR: queries overflow the depth limit!
}
