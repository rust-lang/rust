// Tests that we forbid coercion from `[T; n]` to `&[T]`

fn main() {
    let _: &[i32] = [0];
    //~^ ERROR mismatched types
    //~| expected reference `&[i32]`
    //~| expected &[i32], found array of 1 element
}
