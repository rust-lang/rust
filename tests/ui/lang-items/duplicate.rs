//@ normalize-stderr: "loaded from .*libcore-.*.rlib" -> "loaded from SYSROOT/libcore-*.rlib"
#![feature(lang_items)]

#[lang = "sized"]
trait Sized {}
//~^ ERROR: duplicate lang item

#[lang = "tuple_trait"]
pub trait Tuple {}
// no error
