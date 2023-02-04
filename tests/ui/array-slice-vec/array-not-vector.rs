fn main() {
    let _x: i32 = [1, 2, 3];
    //~^ ERROR mismatched types
    //~| expected `i32`, found `[{integer}; 3]`

    let x: &[i32] = &[1, 2, 3];
    let _y: &i32 = x;
    //~^ ERROR mismatched types
    //~| expected reference `&i32`
    //~| found reference `&[i32]`
    //~| expected `&i32`, found `&[i32]`
}
