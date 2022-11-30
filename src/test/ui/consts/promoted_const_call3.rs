pub const fn id<T>(x: T) -> T { x }
pub const C: () = {
    let _: &'static _ = &String::new();
    //~^ ERROR: destructor of `String` cannot be evaluated at compile-time
    //~| ERROR: temporary value dropped while borrowed

    let _: &'static _ = &id(&String::new());
    //~^ ERROR: destructor of `String` cannot be evaluated at compile-time

    let _: &'static _ = &std::mem::ManuallyDrop::new(String::new());
    // Promoted. bug!
};

fn main() {}
