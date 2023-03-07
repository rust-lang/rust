// Tests that we forbid coercion from `[T; n]` to `&[T]`

fn main() {
    let _: &[i32] = [0];
    //~^ ERROR mismatched types
    //~| expected `&[i32]`, found `[{integer}; 1]`
}
