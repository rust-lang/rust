fn main() {
    let n = 1;
    if 5 == {
    //~^ ERROR this `if` expression is missing a block after the condition
        println!("five");
    }
}
