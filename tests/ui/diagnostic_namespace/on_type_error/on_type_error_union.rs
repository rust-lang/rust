#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "expected union `{Expected}`, found `{Found}`")]
union MyUnion<T> {
    value: std::mem::ManuallyDrop<T>,
}

fn takes_wrapper(_: MyUnion<i32>) {}

fn main() {
    let u1: MyUnion<i32> = 32;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `MyUnion<i32>`, found integer
    //~| NOTE expected union `MyUnion<i32>`, found `{integer}`
    //~| NOTE expected union `MyUnion<i32>`
}
