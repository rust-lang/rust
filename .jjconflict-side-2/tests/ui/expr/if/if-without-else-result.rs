//@ dont-require-annotations: NOTE

fn main() {
    let a = if true { true };
    //~^ ERROR `if` may be missing an `else` clause [E0317]
    //~| NOTE expected `bool`, found `()`
    println!("{}", a);
}
