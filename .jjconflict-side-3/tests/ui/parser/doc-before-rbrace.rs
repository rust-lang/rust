fn main() {
    println!("Hi"); /// hi
    //~^ ERROR found a documentation comment that doesn't document anything
    //~| HELP if a comment was intended use `//`
}
