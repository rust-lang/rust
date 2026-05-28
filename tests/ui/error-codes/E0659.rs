mod moon {
    pub fn foo() {}
}

mod earth {
    pub fn foo() {}
}

mod collider {
    pub use crate::moon::*;
    pub use crate::earth::*;
}

fn main() {
    collider::foo(); //~ ERROR E0659
}
