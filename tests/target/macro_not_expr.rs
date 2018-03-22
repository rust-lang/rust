macro_rules! test {
    ($($t:tt)*) => {};
}

fn main() {
    test!( a : B => c d );
}
