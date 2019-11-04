// normalize-stderr-32bit: "0x" -> "$$PREFIX"
// normalize-stderr-64bit: "0x00000000" -> "$$PREFIX"

#![feature(const_generics, const_compare_raw_pointers)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Const<const P: &'static ()>;

#[repr(C)]
union Transmuter {
    pointer: *const (),
    reference: &'static (),
}

fn main() {
    const A: &'static () = {
        unsafe { Transmuter { pointer: 10 as *const () }.reference }
    };
    const B: &'static () = {
        unsafe { Transmuter { pointer: 11 as *const () }.reference }
    };

    let _: Const<{A}> = Const::<{B}>; //~ mismatched types
    let _: Const<{A}> = Const::<{&()}>; //~ mismatched types
    let _: Const<{A}> = Const::<{A}>;
}
