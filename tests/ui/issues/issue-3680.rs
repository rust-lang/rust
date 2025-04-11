fn main() {
    match None {
        Err(_) => ()
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected enum `Option<_>`
        //~| NOTE_NONVIRAL found enum `Result<_, _>`
        //~| NOTE_NONVIRAL expected `Option<_>`, found `Result<_, _>`
    }
}
