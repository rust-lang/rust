// normalize-stderr-test "loaded from .*libcore-.*.rlib" -> "loaded from SYSROOT/libcore-*.rlib"

#![feature(lang_items)]
#[lang="sized"]
trait Sized { } //~ ERROR found duplicate lang item `sized`

fn ref_Struct(self: &Struct, f: &u32) -> &u32 {
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR cannot find type `Struct` in this scope
    let x = x << 1;
    //~^ ERROR cannot find value `x` in this scope
}

fn main() {}
