// compile-flags:-Z unstable-options --show-coverage
// build-pass (FIXME(62277): could be check-pass?)

//! gotta make sure we can count statics and consts correctly, too

/// static like electricity, right?
pub static THIS_STATIC: usize = 0;

/// (it's not electricity, is it)
pub const THIS_CONST: usize = 1;

/// associated consts show up separately, but let's throw them in as well
pub trait SomeTrait {
    /// just like that, yeah
    const ASSOC_CONST: usize;
}

pub struct SomeStruct;

impl SomeStruct {
    /// wait, structs can have them too, can't forget those
    pub const ASSOC_CONST: usize = 100;
}
