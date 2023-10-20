// skip-filecheck
// unit-test: DataflowConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

#[derive(Copy, Clone)]
struct S(i32);

#[derive(Copy, Clone)]
struct BigStruct(S, u8, f32, S);

// EMIT_MIR struct.main.DataflowConstProp.diff
fn main() {
    let mut s = S(1);
    let a = s.0 + 2;
    s.0 = 3;
    let b = a + s.0;

    const VAL: BigStruct = BigStruct(S(1), 5, 7., S(13));
    let BigStruct(a, b, c, d) = VAL;

    static STAT: &BigStruct = &BigStruct(S(1), 5, 7., S(13));
    let BigStruct(a, b, c, d) = *STAT;
}
