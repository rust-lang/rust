fn main() {
    match None { //~ NOTE this expression has type `Option<_>`
        Err(_) => ()
        //~^ ERROR mismatched types
        //~| NOTE expected enum `Option<_>`
        //~| NOTE found enum `Result<_, _>`
        //~| NOTE expected `Option<_>`, found `Result<_, _>`
    }
}
