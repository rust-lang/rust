//@ run-rustfix

fn main() {
    if true {
        "A".to_string()
    } else {
        "B" //~ ERROR `if` and `else` have incompatible types
    };
}
