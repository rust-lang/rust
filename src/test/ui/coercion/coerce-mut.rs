fn f(x: &mut i32) {}

fn main() {
    let x = 0;
    f(&x);
    //~^ ERROR mismatched types
    //~| expected mutable reference `&'z1 mut i32`
    //~| found reference `&'z2 {integer}`
    //~| types differ in mutability
}
