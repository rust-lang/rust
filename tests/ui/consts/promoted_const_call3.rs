pub const fn id<T>(x: T) -> T { x }
pub const C: () = {
    let _: &'static _ = &String::new();
    //~^ ERROR: destructor of `String` cannot be evaluated at compile-time
};

pub const _: () = {
    let _: &'static _ = &id(&String::new());
    //~^ ERROR: destructor of `String` cannot be evaluated at compile-time
};

pub const _: () = {
    let _: &'static _ = &std::mem::ManuallyDrop::new(String::new());
    //~^ ERROR: temporary value dropped while borrowed
};

fn main() {
    let _: &'static _ = &String::new();
    //~^ ERROR: temporary value dropped while borrowed

    let _: &'static _ = &id(&String::new());
    //~^ ERROR: temporary value dropped while borrowed
    //~| ERROR: temporary value dropped while borrowed

    let _: &'static _ = &std::mem::ManuallyDrop::new(String::new());
    //~^ ERROR: temporary value dropped while borrowed
}
