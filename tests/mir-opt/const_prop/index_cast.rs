//@ test-mir-pass: GVN
// EMIT_MIR_FOR_EACH_BIT_WIDTH

fn main() {}

// EMIT_MIR index_cast.array.GVN.diff
fn array(input: [i32; 256], lit: u8) {
    // CHECK-LABEL: fn array(_1: [i32; 256], _2: u8) -> () {
    // CHECK: debug x => _3;

    // CHECK: assert(const true
    // CHECK: _3 = copy _1[_4];
    let x = input[lit as usize];
}

// EMIT_MIR index_cast.lt.GVN.diff
fn lt(input: u8) {
    // CHECK-LABEL: fn lt(_1: u8) -> () {

    // CHECK: debug yes => _2;
    // CHECK: debug no => _5;

    // CHECK: _2 = const true;
    let yes = 256u64 > (input as u64);

    // CHECK: _5 = const false;
    let no = (input as u64) > 256u64;
}

// EMIT_MIR index_cast.le.GVN.diff
fn le(input: u16) {
    // CHECK-LABEL: fn le(_1: u16) -> () {

    // CHECK: debug yes => _2;
    // CHECK: debug runtime => _5;
    // CHECK: debug no => _9;

    // CHECK: _2 = const true;
    let yes = (input as u32) <= U16_MAX_PLUS_ONE;

    // CHECK: _5 = Le(const 65535_u32, copy _3);
    let runtime = u16::MAX as u32 <= (input as u32);

    // CHECK: _9 = const false;
    let no = U16_MAX_PLUS_ONE <= (input as u32);
}

// EMIT_MIR index_cast.gt.GVN.diff
fn gt(input: u8) {
    // CHECK-LABEL: fn gt(_1: u8) -> () {

    // CHECK: debug yes => _2;
    // CHECK: debug no => _5;

    // CHECK: _2 = const true;
    let yes = 256u16 > (input as u16);

    // CHECK: _5 = const false;
    let no = (input as u16) > 256u16;
}

// EMIT_MIR index_cast.ge.GVN.diff
fn ge(input: u32) {
    // CHECK-LABEL: fn ge(_1: u32) -> () {

    // CHECK: debug yes => _2;
    // CHECK: debug runtime => _5;
    // CHECK: debug no => _9;

    // CHECK: _2 = const true;
    let yes = U32_MAX_PLUS_ONE >= (input as u64);

    // CHECK: _5 = Ge(copy _3
    let runtime = (input as u64) >= u32::MAX as u64;

    // CHECK: _9 = const false;
    let no = (input as u64) >= U32_MAX_PLUS_ONE;
}

const U16_MAX_PLUS_ONE: u32 = (u16::MAX as u32) + 1u32;
const U32_MAX_PLUS_ONE: u64 = (u32::MAX as u64) + 1u64;
