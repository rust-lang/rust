fn main() {
    let a = if true { true };
    //~^ ERROR `if` may be missing an `else` clause [E0317]
    //~| NOTE_NONVIRAL expected `bool`, found `()`
    println!("{}", a);
}
