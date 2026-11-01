fn main() {
    let x = if true {
        &true
    } else if false {
        true //~ ERROR `if` and `else` have incompatible types [E0308]
        //~^ HELP consider borrowing here
    } else {
        true
    };
}
