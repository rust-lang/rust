// https://github.com/rust-lang/rust/issues/91489
//@ check-pass

// regression test for #91489

use std::borrow::Borrow;
use std::borrow::Cow;

pub struct VariantType {}
pub struct VariantTy {}

impl Borrow<VariantTy> for VariantType {
    fn borrow(&self) -> &VariantTy {
        unimplemented!()
    }
}

impl ToOwned for VariantTy {
    type Owned = VariantType;
    fn to_owned(&self) -> VariantType {
        unimplemented!()
    }
}

impl VariantTy {
    pub fn as_str(&self) -> () {}
}

// the presence of this was causing all attempts to call `as_str` on
// `Cow<'_, VariantTy>, including in itself, to not find the method
static _TYP: () = {
    let _ = || {
        // should be found
        Cow::Borrowed(&VariantTy {}).as_str();
    };
};

fn main() {
    // should be found
    Cow::Borrowed(&VariantTy {}).as_str()
}
