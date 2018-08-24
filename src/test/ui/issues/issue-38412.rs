fn main() {
    let Box(a) = loop { };
    //~^ ERROR expected tuple struct/variant, found struct `Box`

    // (The below is a trick to allow compiler to infer a type for
    // variable `a` without attempting to ascribe a type to the
    // pattern or otherwise attempting to name the Box type, which
    // would run afoul of issue #22207)
    let _b: *mut i32 = *a;
}
