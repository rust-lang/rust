#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
fn main() {
    println!("Hello, World!");
    main();
    //~^ main_recursion
}
