#![feature(rustc_attrs)]
#![feature(staged_api)]
#![stable(feature = "a", since = "1.0.0")]

#[rustc_promotable]
#[stable(feature = "a", since = "1.0.0")]
#[rustc_const_stable(feature = "a", since = "1.0.0")]
pub const fn id<T>(x: &'static T) -> &'static T { x }

#[rustc_promotable]
#[stable(feature = "a", since = "1.0.0")]
#[rustc_const_stable(feature = "a", since = "1.0.0")]
pub const fn new_string() -> String {
    String::new()
}

#[rustc_promotable]
#[stable(feature = "a", since = "1.0.0")]
#[rustc_const_stable(feature = "a", since = "1.0.0")]
pub const fn new_manually_drop<T>(t: T) -> std::mem::ManuallyDrop<T>  {
    std::mem::ManuallyDrop::new(t)
}


const C: () = {
    let _: &'static _ = &id(&new_string());
    //~^ ERROR destructor of `String` cannot be evaluated at compile-time
};

const _: () = {
    let _: &'static _ = &new_manually_drop(new_string());
    //~^ ERROR: temporary value dropped while borrowed
};

fn main() {
    let _: &'static _ = &id(&new_string());
    //~^ ERROR: temporary value dropped while borrowed
    //~| ERROR: temporary value dropped while borrowed

    let _: &'static _ = &new_manually_drop(new_string());
    //~^ ERROR: temporary value dropped while borrowed
}
