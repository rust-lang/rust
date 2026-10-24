//@ dont-require-annotations: NOTE

fn f(x: &mut i32) {}

fn main() {
    let x = 0;
    f(&x);
    //~^ ERROR mismatched types
    //~| NOTE expected mutable reference `&mut i32`
    //~| NOTE found reference `&{integer}`
    //~| NOTE types differ in mutability
}
