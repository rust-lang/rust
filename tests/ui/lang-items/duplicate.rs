//@ normalize-stderr: "loaded from .*libcore-.*.rmeta" -> "loaded from SYSROOT/libcore-*.rmeta"
#![feature(lang_items)]

#[lang = "sized"]
trait Sized {}
//~^ ERROR: duplicate lang item

#[lang = "tuple_trait"]
pub trait Tuple {}
// no error
