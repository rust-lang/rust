#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
fn main() {
    println!("Hello, World!");
    main();
    //~^ ERROR: recursing into entrypoint `main`
}
