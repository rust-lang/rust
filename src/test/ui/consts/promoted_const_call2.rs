#![feature(const_precise_live_drops)]
pub const fn id<T>(x: T) -> T { x }
pub const C: () = {
    let _: &'static _ = &id(&String::new());
    //~^ ERROR: temporary value dropped while borrowed
    //~| ERROR: temporary value dropped while borrowed
    //~| ERROR: destructor of `String` cannot be evaluated at compile-time
};

fn main() {
    let _: &'static _ = &id(&String::new());
    //~^ ERROR: temporary value dropped while borrowed
    //~| ERROR: temporary value dropped while borrowed
}
