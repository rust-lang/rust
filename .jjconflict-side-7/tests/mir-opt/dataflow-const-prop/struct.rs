//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes
// EMIT_MIR_FOR_EACH_BIT_WIDTH

#[derive(Copy, Clone)]
struct S(i32);

#[derive(Copy, Clone)]
struct SmallStruct(f32, Option<S>, &'static [f32]);

#[derive(Copy, Clone)]
struct BigStruct(f32, Option<S>, &'static [f32]);

// EMIT_MIR struct.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug s => [[s:_.*]];
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: debug a1 => [[a1:_.*]];
    // CHECK: debug b1 => [[b1:_.*]];
    // CHECK: debug c1 => [[c1:_.*]];
    // CHECK: debug a2 => [[a2:_.*]];
    // CHECK: debug b2 => [[b2:_.*]];
    // CHECK: debug c2 => [[c2:_.*]];
    // CHECK: debug ss => [[ss:_.*]];
    // CHECK: debug a3 => [[a3:_.*]];
    // CHECK: debug b3 => [[b3:_.*]];
    // CHECK: debug c3 => [[c3:_.*]];
    // CHECK: debug a4 => [[a4:_.*]];
    // CHECK: debug b4 => [[b4:_.*]];
    // CHECK: debug c4 => [[c4:_.*]];
    // CHECK: debug bs => [[bs:_.*]];

    // CHECK: [[s]] = const S(1_i32);
    let mut s = S(1);

    // CHECK: [[a]] = const 3_i32;
    let a = s.0 + 2;
    s.0 = 3;

    // CHECK: [[b]] = const 6_i32;
    let b = a + s.0;

    const SMALL_VAL: SmallStruct = SmallStruct(4., Some(S(1)), &[]);

    // CHECK: [[a1]] = const 4f32;
    // CHECK: [[b1]] = copy ({{_.*}}.1: std::option::Option<S>);
    // CHECK: [[c1]] = copy ({{_.*}}.2: &[f32]);
    let SmallStruct(a1, b1, c1) = SMALL_VAL;

    static SMALL_STAT: &SmallStruct = &SmallStruct(9., None, &[13.]);

    // CHECK: [[a2]] = const 9f32;
    // CHECK: [[b2]] = copy ((*{{_.*}}).1: std::option::Option<S>);
    // CHECK: [[c2]] = copy ((*{{_.*}}).2: &[f32]);
    let SmallStruct(a2, b2, c2) = *SMALL_STAT;

    // CHECK: [[ss]] = SmallStruct(const 9f32, move {{_.*}}, move {{_.*}});
    let ss = SmallStruct(a2, b2, c2);

    const BIG_VAL: BigStruct = BigStruct(25., None, &[]);

    // CHECK: [[a3]] = const 25f32;
    // CHECK: [[b3]] = copy ({{_.*}}.1: std::option::Option<S>);
    // CHECK: [[c3]] = copy ({{_.*}}.2: &[f32]);
    let BigStruct(a3, b3, c3) = BIG_VAL;

    static BIG_STAT: &BigStruct = &BigStruct(82., Some(S(35)), &[45., 72.]);
    // CHECK: [[a4]] = const 82f32;
    // CHECK: [[b4]] = copy ((*{{_.*}}).1: std::option::Option<S>);
    // CHECK: [[c4]] = copy ((*{{_.*}}).2: &[f32]);
    let BigStruct(a4, b4, c4) = *BIG_STAT;

    // We arbitrarily limit the size of synthetized values to 4 pointers.
    // `BigStruct` can be read, but we will keep a MIR aggregate for this.
    // CHECK: [[bs]] = BigStruct(const 82f32, move {{.*}}, move {{_.*}});
    let bs = BigStruct(a4, b4, c4);
}
