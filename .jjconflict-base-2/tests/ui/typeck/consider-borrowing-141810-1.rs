fn main() {
    let x = if true {
        &true
    } else if false { //~ ERROR `if` and `else` have incompatible types [E0308]
        true //~ HELP consider borrowing here
    } else {
        true
    };
}
