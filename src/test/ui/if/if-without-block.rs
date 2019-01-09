fn main() {
    let n = 1;
    if 5 == {
    //~^ NOTE this `if` statement has a condition, but no block
        println!("five");
    }
}
//~^ ERROR expected `{`, found `}`
//~| NOTE expected `{`
