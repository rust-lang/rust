#[warn(clippy::main_recursion)]
#[allow(unconditional_recursion)]
fn main() {
    main();
}
