//! Removing a private import that triggers `hidden_glob_reexports` can
//! break upstream code!
//!
//! Also see <https://github.com/rust-lang/rust/issues/159901>

//@revisions: keep remove
//@[keep]check-pass

#![crate_type = "lib"]

use rustc_span::Ident;

pub fn foo() {
    use rustc_ast::*; // Must be inside the function scope
    let _ /* fn(_,_) -> _*/ = Ident;
    Ident::x();
    //[remove]~^ ERROR cannot find module `Ident` in this scope
}

pub mod rustc_ast {
    pub use self::TokenKind::*;

    #[cfg(keep)] // Compile error if false
    #[expect(hidden_glob_reexports, unused_imports)]
    use super::rustc_span::Ident;

    pub enum TokenKind {
        Ident(u8, u8),
    }
}

pub mod rustc_span {
    pub struct Ident {}

    impl Ident {
        pub fn x() -> bool {
            true
        }
    }
}
