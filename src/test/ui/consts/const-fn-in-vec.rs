fn main() {
    // should hint to create an inline `const` block
    // or to create a new `const` item
    let strings: [String; 5] = [String::new(); 5];
    //~^ ERROR the trait bound `String: Copy` is not satisfied
    println!("{:?}", strings);
}
