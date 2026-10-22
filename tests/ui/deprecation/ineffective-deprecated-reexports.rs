//! `#[deprecated]` on a `use` item is accepted but has no effect (#30827), so it warns (#142436).
//@ check-pass

#![allow(unused_imports)]

mod inner {
    pub struct A;
    pub struct B;
    pub fn f() {}
    pub mod m {
        pub struct C;
    }
    #[deprecated]
    pub struct D;
}

#[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use inner::A;

#[deprecated = "use `inner::B` directly"] //~ WARN `#[deprecated]` on a `use` item has no effect
use inner::B as PrivateB;

#[deprecated(since = "1.0.0", note = "moved")] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use inner::{f, B};

#[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use inner::{m::{self, C}, A as A2};

#[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use inner::m as module;

mod globbed {
    #[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
    pub use crate::inner::*;
}

fn body() {
    #[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
    use crate::inner::A as Local;
}

macro_rules! reexport {
    ($name:ident) => {
        #[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
        pub use crate::inner::A as $name;
    };
}
reexport!(FromMacro);

// The lint is about the attribute on the `use` item; a deprecated target still warns on its own.
#[deprecated] //~ WARN `#[deprecated]` on a `use` item has no effect
pub use inner::D as DeprecatedTarget; //~ WARN use of deprecated unit struct

// Only the `use` item's own attribute counts: an import inside a deprecated module is silent.
#[deprecated]
mod deprecated_mod {
    pub use crate::inner::A as InDeprecatedMod;
}

#[expect(ineffective_deprecated_reexports)]
#[deprecated]
pub use inner::A as ExpectedA;

#[expect(ineffective_deprecated_reexports)]
#[deprecated]
pub use inner::{A as ExpectedGroupA, B as ExpectedGroupB};

#[allow(ineffective_deprecated_reexports)]
#[deprecated]
pub use inner::A as AllowedA;

fn main() {}
