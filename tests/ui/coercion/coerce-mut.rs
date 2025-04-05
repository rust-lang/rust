fn f(x: &mut i32) {}

fn main() {
    let x = 0;
    f(&x);
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected mutable reference `&mut i32`
    //~| NOTE_NONVIRAL found reference `&{integer}`
    //~| NOTE_NONVIRAL types differ in mutability
}
