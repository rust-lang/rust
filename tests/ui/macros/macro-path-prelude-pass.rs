//@ check-pass

mod m {
    fn check() {
        std::panic!(); // OK
    }
}

fn main() {}
