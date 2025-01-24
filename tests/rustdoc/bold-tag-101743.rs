// Regression test for https://github.com/rust-lang/rust/issues/101743

#![crate_name="foo"]

pub type Word = usize;
pub struct Repr<const B: usize>([i32; B]);
pub struct IBig(usize);

pub const fn base_as_ibig<const B: Word>() -> IBig {
    IBig(B)
}

impl<const B: Word> Repr<B> {
    // If we change back to rendering the value of consts, check this doesn't add
    // a <b> tag, but escapes correctly

    //@ !has foo/struct.Repr.html '//section[@id="associatedconstant.BASE"]/h4' '='
    pub const BASE: IBig = base_as_ibig::<B>();
}
