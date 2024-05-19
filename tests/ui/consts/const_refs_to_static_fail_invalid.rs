//@ normalize-stderr-test "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr-test "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"
#![feature(const_refs_to_static)]
#![allow(static_mut_refs)]

fn invalid() {
    static S: i8 = 10;

    const C: &bool = unsafe { std::mem::transmute(&S) };
    //~^ERROR: undefined behavior
    //~| expected a boolean

    // This must be rejected here (or earlier), since it's not a valid `&bool`.
    match &true {
        C => {} //~ERROR: could not evaluate constant pattern
        _ => {}
    }
}

fn extern_() {
    extern "C" {
        static S: i8;
    }

    const C: &i8 = unsafe { &S };
    //~^ERROR: undefined behavior
    //~| `extern` static

    // This must be rejected here (or earlier), since the pattern cannot be read.
    match &0 {
        C => {} //~ERROR: could not evaluate constant pattern
        _ => {}
    }
}

fn mutable() {
    static mut S_MUT: i32 = 0;

    const C: &i32 = unsafe { &S_MUT };
    //~^ERROR: undefined behavior
    //~| encountered reference to mutable memory

    // This *must not build*, the constant we are matching against
    // could change its value!
    match &42 {
        C => {} //~ERROR: could not evaluate constant pattern
        _ => {}
    }
}

fn main() {}
