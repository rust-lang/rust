// error-pattern: mismatched types

tag a { A(int); }
tag b { B(int); }

fn main() {
    let a x = A(0);
    alt (x) {
        case (B(?y)) {}
    }
}

