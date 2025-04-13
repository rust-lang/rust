//@ dont-require-annotations: NOTE

fn main() {
    let _x: i32 = [1, 2, 3];
    //~^ ERROR mismatched types
    //~| NOTE expected `i32`, found `[{integer}; 3]`

    let x: &[i32] = &[1, 2, 3];
    let _y: &i32 = x;
    //~^ ERROR mismatched types
    //~| NOTE expected reference `&i32`
    //~| NOTE found reference `&[i32]`
    //~| NOTE expected `&i32`, found `&[i32]`
}
