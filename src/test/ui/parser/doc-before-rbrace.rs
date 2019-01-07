fn main() {
    println!("Hi"); /// hi
    //~^ ERROR found a documentation comment that doesn't document anything
    //~| HELP maybe a comment was intended
}
