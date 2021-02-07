fn f(x: &mut i32) {}

fn main() {
    let x = 0;
    f(&x);
    //~^ ERROR arguments to this function are incorrect
    //~| expected mutable reference `&mut i32`
    //~| found reference `&{integer}`
    //~| types differ in mutability
}
