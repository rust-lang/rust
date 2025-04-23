fn main() {
    match true {
        _ if let true = true && true => {}
        //~^ ERROR `if let` guards are experimental
        //~| ERROR `let` expressions in this position are unstable
        _ => {}
    }
}
