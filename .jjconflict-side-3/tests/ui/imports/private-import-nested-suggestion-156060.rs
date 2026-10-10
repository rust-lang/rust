// Regression test for #156060.

mod one {
    pub struct One();
}

mod two {
    use crate::one::One;
    pub struct Two();
}

mod test {
    use crate::two::{One, Two};
    //~^ ERROR struct import `One` is private [E0603]
}

fn main() {}
