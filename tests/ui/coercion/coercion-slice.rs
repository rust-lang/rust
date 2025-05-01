// Tests that we forbid coercion from `[T; n]` to `&[T]`

fn main() {
    let _: &[i32] = [0];
    //~^ ERROR mismatched types
    //~| NOTE expected `&[i32]`, found `[{integer}; 1]`
    //~| NOTE expected due to this
}
