// The gate for `default impl` is exercised in `defaultimpl/specialization-feature-gate-default.rs`.

trait Trait {
    type Ty;
    const CT: ();
    fn fn_(&self);
}

impl<T> Trait for T {
    default type Ty = (); //~ ERROR specialization is experimental
    default const CT: () = (); //~ ERROR specialization is experimental
    default fn fn_(&self) {} //~ ERROR specialization is experimental
}

fn main() {}
