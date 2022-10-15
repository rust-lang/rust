// normalize-stderr-test "loaded from .*libcore-.*.rlib" -> "loaded from SYSROOT/libcore-*.rlib"

#![feature(lang_items)]
#[lang="sized"]
trait Sized { } //~ ERROR found duplicate lang item `sized`

fn ref_Struct(self: &Struct, f: &u32) -> &u32 {
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR cannot find type `Struct` in this scope
    //~| ERROR mismatched types
    let x = x << 1;
    //~^ ERROR the size for values of type `{integer}` cannot be known at compilation time
    //~| ERROR cannot find value `x` in this scope
}

fn main() {}
