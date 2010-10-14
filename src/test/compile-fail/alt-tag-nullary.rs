// error-pattern: mismatched types

tag a { A; }
tag b { B; }

fn main() {
    let a x = A;
    alt (x) {
        case (B) {}
    }
}

