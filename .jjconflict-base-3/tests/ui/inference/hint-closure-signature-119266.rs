fn main() {
    let x = |a: u8, b: (usize, u32), c: fn() -> char| -> String { "I love beans.".to_string() };
    //~^ NOTE: the found closure

    let x: fn(i32) = x;
    //~^ ERROR: mismatched types [E0308]
    //~| NOTE: incorrect number of function parameters
    //~| NOTE: expected due to this
    //~| NOTE: expected fn pointer `fn(i32)`
    //~| NOTE: closure has signature: `fn(u8, (usize, u32), fn() -> char) -> String`
}
