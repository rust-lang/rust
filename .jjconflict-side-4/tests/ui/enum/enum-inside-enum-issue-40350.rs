//@ check-pass

enum E {
    A = {
        enum F { B }
        0
    }
}

fn main() {}
