// skip-filecheck
//@ test-mir-pass: GVN
//@ ignore-endian-big
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR const_allocation2.main.GVN.after.mir
fn main() {
    FOO;
}

const BAR: [u8; 4] = [42, 69, 21, 111];

static FOO: &[(Option<i32>, &[&u8])] =
    &[(None, &[]), (None, &[&5, &6]), (Some(42), &[&BAR[3], &42, &BAR[2]])];
