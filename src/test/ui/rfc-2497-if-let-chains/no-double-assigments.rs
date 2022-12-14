// check-pass

fn main() {
    loop {
        // [1][0] should leave top scope
        if true && [1][0] == 1 && true {
        }
    }
}
