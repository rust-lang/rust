fn f(x: &mut i32) {}

fn main() {
    let x = 0;
    f(&x);
    //~^ ERROR mismatched types
    //~| expected type `&mut i32`
    //~| found type `&{integer}`
    //~| types differ in mutability
}
