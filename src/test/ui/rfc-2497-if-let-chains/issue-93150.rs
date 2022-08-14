fn main() {
    match true {
        _ if let true = true && true => {}
        //~^ ERROR `if let` guards are
        //~| ERROR `let` expressions in this
        _ => {}
    }
}
