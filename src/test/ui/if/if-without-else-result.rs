fn main() {
    let a = if true { true };
    //~^ ERROR if may be missing an else clause [E0317]
    //~| expected unit type `()`
    //~| found type `bool`
    //~| expected (), found bool
    println!("{}", a);
}
