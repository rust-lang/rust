//@ check-pass

#[derive(Debug)]
struct H;

mod p {
    use super::*;

    #[derive(Clone)]
    struct H;

    mod t {
        use super::*;

        fn f() {
           let h: crate::p::H = H;
        }
    }
}

fn main() {}
