#![crate_type = "lib"]

static private: isize = 0;
pub static public: isize = 0;

pub struct A(());

impl A {
    fn foo() {}
}

mod foo {
    pub static a: isize = 0;
    pub fn b() {}
    pub struct c;
    pub enum d {}
    pub type e = isize;

    pub struct A(());

    impl A {
        fn foo() {}
    }

    // these are public so the parent can re-export them.
    pub static reexported_a: isize = 0;
    pub fn reexported_b() {}
    pub struct reexported_c;
    pub enum reexported_d {}
    pub type reexported_e = isize;
}

pub mod bar {
    pub use crate::foo::reexported_a as e;
    pub use crate::foo::reexported_b as f;
    pub use crate::foo::reexported_c as g;
    pub use crate::foo::reexported_d as h;
    pub use crate::foo::reexported_e as i;
}

pub static a: isize = 0;
pub fn b() {}
pub struct c;
pub enum d {}
pub type e = isize;

static j: isize = 0;
fn k() {}
struct l;
enum m {}
type n = isize;
