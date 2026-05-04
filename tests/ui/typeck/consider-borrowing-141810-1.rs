fn main() {
    let x = if true {
        &true
    } else if false {
        true //~ HELP consider borrowing here
        //~^ ERROR `if` and `else` have incompatible types [E0308]
    } else {
        true
    };
}
