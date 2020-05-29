// compile-flags: -Z save-analysis

fn main() {
    let _: Box<(dyn ?Sized)>;
    //~^ ERROR `?Trait` is not permitted in trait object types
    //~| ERROR at least one trait is required for an object type
}
