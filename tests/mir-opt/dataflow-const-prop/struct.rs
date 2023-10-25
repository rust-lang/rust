// skip-filecheck
// unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

#[derive(Copy, Clone)]
struct S(i32);

#[derive(Copy, Clone)]
struct SmallStruct(f32, Option<S>, &'static [f32]);

#[derive(Copy, Clone)]
struct BigStruct(f32, Option<S>, &'static [f32]);

// EMIT_MIR struct.main.DataflowConstProp.diff
fn main() {
    let mut s = S(1);
    let a = s.0 + 2;
    s.0 = 3;
    let b = a + s.0;

    const SMALL_VAL: SmallStruct = SmallStruct(4., Some(S(1)), &[]);
    let SmallStruct(a, b, c) = SMALL_VAL;

    static SMALL_STAT: &SmallStruct = &SmallStruct(9., None, &[13.]);
    let SmallStruct(a, b, c) = *SMALL_STAT;

    let ss = SmallStruct(a, b, c);

    const BIG_VAL: BigStruct = BigStruct(25., None, &[]);
    let BigStruct(a, b, c) = BIG_VAL;

    static BIG_STAT: &BigStruct = &BigStruct(82., Some(S(35)), &[45., 72.]);
    let BigStruct(a, b, c) = *BIG_STAT;

    // We arbitrarily limit the size of synthetized values to 4 pointers.
    // `BigStruct` can be read, but we will keep a MIR aggregate for this.
    let bs = BigStruct(a, b, c);
}
