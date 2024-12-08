mod A {
    pub type B = ();
    pub type B2 = ();
}

mod C {
    use crate::D::B as _;
    //~^ ERROR unresolved import `crate::D::B`

    use crate::D::B2;
    //~^ ERROR unresolved import `crate::D::B2`
}

mod D {}

fn main() {}
