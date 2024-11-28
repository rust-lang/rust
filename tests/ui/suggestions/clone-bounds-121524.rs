#[derive(Clone)]
struct ThingThatDoesAThing;

trait DoesAThing {}

impl DoesAThing for ThingThatDoesAThing {}

fn clones_impl_ref_inline(thing: &impl DoesAThing) {
    //~^ HELP consider restricting opaque type `impl DoesAThing` with trait `Clone`
    drops_impl_owned(thing.clone()); //~ ERROR E0277
    //~^ NOTE copies the reference
    //~| NOTE the trait `DoesAThing` is not implemented for `&impl DoesAThing`
}

fn drops_impl_owned(_thing: impl DoesAThing) { }

fn main() {
    clones_impl_ref_inline(&ThingThatDoesAThing);
}
