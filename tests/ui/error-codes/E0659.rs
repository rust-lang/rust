mod moon {
    pub fn foo() {}
}

mod earth {
    pub fn foo() {}
}

mod collider {
    pub use moon::*;
    pub use earth::*;
}

fn main() {
    collider::foo(); //~ ERROR E0659
}
