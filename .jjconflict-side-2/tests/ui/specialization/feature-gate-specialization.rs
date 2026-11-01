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

trait OtherTrait {
    fn fn_();
}

default impl<T> OtherTrait for T { //~ ERROR specialization is experimental
    fn fn_() {}
}

fn main() {}
