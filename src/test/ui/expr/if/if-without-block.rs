fn main() {
    let n = 1;
    if 5 == {
    //~^ NOTE this `if` expression has a condition, but no block
        println!("five");
    }
}
//~^ ERROR expected `{`, found `}`
//~| NOTE expected `{`
