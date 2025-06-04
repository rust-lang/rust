//@ check-pass

#![feature(decl_macro)]

mod type_ns {
    pub type A = u8;
}
mod value_ns {
    pub const A: u8 = 0;
}
mod macro_ns {
    pub macro A() {}
}

mod merge2 {
    pub use crate::type_ns::A;
    pub use crate::value_ns::A;
}
mod merge3 {
    pub use crate::type_ns::A;
    pub use crate::value_ns::A;
    pub use crate::macro_ns::A;
}

mod use2 {
    pub use crate::merge2::A;
}
mod use3 {
    pub use crate::merge3::A;
}

fn main() {
    type B2 = use2::A;
    let a2 = use2::A;

    type B3 = use3::A;
    let a3 = use3::A;
    use3::A!();
}
