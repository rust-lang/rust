struct boxed_int {
    f: &int,
}

fn get(bi: &r/boxed_int) -> &r/int {
    bi.f
}

fn with(bi: &r/boxed_int) {
    // Here, the upcast is allowed because the `boxed_int` type is
    // contravariant with respect to `&r`.  See also
    // compile-fail/regions-infer-invariance-due-to-mutability.rs
    let bi: &blk/boxed_int/&blk = bi;
    assert *get(bi) == 22;
}

fn main() {
    let g = 22;
    let foo = boxed_int { f: &g };
    with(&foo);
}