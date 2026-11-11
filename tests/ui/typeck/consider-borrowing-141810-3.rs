fn main() {
    let x = if true {
        &()
    } else if false { //~ ERROR `if` and `else` have incompatible types [E0308]

    };
}
