//! Regression test for an LLVM assertion that used to be hit when:
//!
//! - There's a generic enum contained within a tuple struct
//! - When the tuple struct is parameterized by some lifetime `'a`
//! - The enum is concretized with its type argument being a reference to a trait object (of
//!   lifetime `'a`)
//!
//! Issue: <https://github.com/rust-lang/rust/issues/9719>

//@ build-pass

// Dummy trait implemented for `isize` to use in the test cases
pub trait MyTrait {
    fn dummy(&self) {}
}
impl MyTrait for isize {}

// `&dyn MyTrait` contained in enum variant
pub struct EnumRefDynTrait<'a>(Enum<&'a (dyn MyTrait + 'a)>);
pub enum Enum<T> {
    Variant(T),
}

fn enum_dyn_trait() {
    let x: isize = 42;
    let y = EnumRefDynTrait(Enum::Variant(&x as &dyn MyTrait));
    let _ = y;
}

// `&dyn MyTrait` contained behind `Option` in named field of struct
struct RefDynTraitNamed<'a> {
    x: Option<&'a (dyn MyTrait + 'a)>,
}

fn named_option_dyn_trait() {
    let x: isize = 42;
    let y = RefDynTraitNamed { x: Some(&x as &dyn MyTrait) };
    let _ = y;
}

// `&dyn MyTrait` contained behind `Option` in unnamed field of struct
pub struct RefDynTraitUnnamed<'a>(Option<&'a (dyn MyTrait + 'a)>);

fn unnamed_option_dyn_trait() {
    let x: isize = 42;
    let y = RefDynTraitUnnamed(Some(&x as &dyn MyTrait));
    let _ = y;
}

pub fn main() {
    enum_dyn_trait();
    named_option_dyn_trait();
    unnamed_option_dyn_trait();
}
