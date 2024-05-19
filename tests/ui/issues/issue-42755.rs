macro_rules! foo {
    ($($p:vis)*) => {} //~ ERROR repetition matches empty token tree
}

foo!(a);

fn main() {}
