#![feature(extern_types)]

trait Trait {
    #[inline] //~ ERROR attribute should be applied to function or closure
    const X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;
}

extern {
    #[inline] //~ ERROR attribute should be applied to function or closure
    static X: u32;

    #[inline] //~ ERROR attribute should be applied to function or closure
    type T;
}

fn main() {}
