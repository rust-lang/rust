//@ check-pass

mod a {
    mod b {
        use self as A;
        use super as B;
        use super::{self as C};
    }
}

fn main() {}
