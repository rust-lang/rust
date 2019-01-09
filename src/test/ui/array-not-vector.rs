fn main() {
    let _x: i32 = [1, 2, 3];
    //~^ ERROR mismatched types
    //~| expected type `i32`
    //~| found type `[{integer}; 3]`
    //~| expected i32, found array of 3 elements

    let x: &[i32] = &[1, 2, 3];
    let _y: &i32 = x;
    //~^ ERROR mismatched types
    //~| expected type `&i32`
    //~| found type `&[i32]`
    //~| expected i32, found slice
}
