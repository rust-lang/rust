fn main() {
    let x: i32 = 5;
    let ptr: *const i32 = &x;
    let dptr: **const i32 = &ptr;
    //~^ ERROR expected `mut` or `const` keyword in raw pointer type
    //~| HELP add `mut` or `const` here
}
