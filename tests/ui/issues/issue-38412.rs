fn main() {
    let Box(a) = loop { };
    //~^ ERROR cannot match against a tuple struct which contains private fields

    // (The below is a trick to allow compiler to infer a type for
    // variable `a` without attempting to ascribe a type to the
    // pattern or otherwise attempting to name the Box type, which
    // would run afoul of issue #22207)
    let _b: *mut i32 = *a;
}
