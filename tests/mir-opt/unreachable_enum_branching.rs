//@ test-mir-pass: UnreachableEnumBranching
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

enum Empty {}

// test matching an enum with uninhabited variants
enum Test1 {
    A(Empty),
    B(Empty),
    C,
}

// test an enum where the discriminants don't match the variant indexes
// (the optimization should do nothing here)
enum Test2 {
    D = 4,
    E = 5,
}

// test matching an enum with uninhabited variants and multiple inhabited
enum Test3 {
    A(Empty),
    B(Empty),
    C,
    D,
}

enum Test4 {
    A(i32),
    B(i32),
    C,
    D,
}

#[repr(i8)]
enum Test5<T> {
    A(T) = -1,
    B(T) = 0,
    C = 5,
    D = 3,
}

struct Plop {
    xx: u32,
    test3: Test3,
}

// EMIT_MIR unreachable_enum_branching.simple.UnreachableEnumBranching.diff
fn simple() {
    // CHECK-LABEL: fn simple(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: [[unreachable:bb.*]], 1: [[unreachable]], 2: bb1, otherwise: [[unreachable]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test1::C {
        Test1::A(_) => "A(Empty)",
        Test1::B(_) => "B(Empty)",
        Test1::C => "C",
    };
}

// EMIT_MIR unreachable_enum_branching.custom_discriminant.UnreachableEnumBranching.diff
fn custom_discriminant() {
    // CHECK-LABEL: fn custom_discriminant(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [4: bb3, 5: bb2, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test2::D {
        Test2::D => "D",
        Test2::E => "E",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t1.UnreachableEnumBranching.diff
fn otherwise_t1() {
    // CHECK-LABEL: fn otherwise_t1(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: bb5, 1: bb5, 2: bb1, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test1::C {
        Test1::A(_) => "A(Empty)",
        Test1::B(_) => "B(Empty)",
        _ => "C",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t2.UnreachableEnumBranching.diff
fn otherwise_t2() {
    // CHECK-LABEL: fn otherwise_t2(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [4: bb2, 5: bb1, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test2::D {
        Test2::D => "D",
        _ => "E",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t3.UnreachableEnumBranching.diff
fn otherwise_t3() {
    // CHECK-LABEL: fn otherwise_t3(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: bb5, 1: bb5, otherwise: bb1];
    // CHECK: bb1: {
    // CHECK-NOT: unreachable;
    // CHECK: }
    // CHECK: bb5: {
    // CHECK-NEXT: unreachable;
    match Test3::C {
        Test3::A(_) => "A(Empty)",
        Test3::B(_) => "B(Empty)",
        _ => "C",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t4_unreachable_default.UnreachableEnumBranching.diff
fn otherwise_t4_unreachable_default() {
    // CHECK-LABEL: fn otherwise_t4_unreachable_default(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: {{bb.*}}, 1: {{bb.*}}, 2: {{bb.*}}, 3: {{bb.*}}, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test4::C {
        Test4::A(_) => "A(i32)",
        Test4::B(_) => "B(i32)",
        Test4::C => "C",
        _ => "D",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t4_unreachable_default_2.UnreachableEnumBranching.diff
fn otherwise_t4_unreachable_default_2() {
    // CHECK-LABEL: fn otherwise_t4_unreachable_default_2(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: {{bb.*}}, 1: {{bb.*}}, 2: {{bb.*}}, 3: {{bb.*}}, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test4::C {
        Test4::A(1) => "A(1)",
        Test4::A(2) => "A(2)",
        Test4::B(_) => "B(i32)",
        Test4::C => "C",
        _ => "A(other)D",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t4.UnreachableEnumBranching.diff
fn otherwise_t4() {
    // CHECK-LABEL: fn otherwise_t4(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: {{bb.*}}, 1: {{bb.*}}, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NOT: unreachable;
    // CHECK: }
    match Test4::C {
        Test4::A(_) => "A(i32)",
        Test4::B(_) => "B(i32)",
        _ => "CD",
    };
}

// EMIT_MIR unreachable_enum_branching.otherwise_t5_unreachable_default.UnreachableEnumBranching.diff
fn otherwise_t5_unreachable_default<T>() {
    // CHECK-LABEL: fn otherwise_t5_unreachable_default(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [255: {{bb.*}}, 0: {{bb.*}}, 5: {{bb.*}}, 3: {{bb.*}}, otherwise: [[unreachable:bb.*]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test5::<T>::C {
        Test5::A(_) => "A(T)",
        Test5::B(_) => "B(T)",
        Test5::C => "C",
        _ => "D",
    };
}

// EMIT_MIR unreachable_enum_branching.byref.UnreachableEnumBranching.diff
fn byref() {
    // CHECK-LABEL: fn byref(
    let plop = Plop { xx: 51, test3: Test3::C };

    // CHECK: [[ref_discr:_.*]] = discriminant((*
    // CHECK: switchInt(move [[ref_discr]]) -> [0: [[unreachable:bb.*]], 1: [[unreachable]], 2: {{bb.*}}, 3: {{bb.*}}, otherwise: [[unreachable]]];
    match &plop.test3 {
        Test3::A(_) => "A(Empty)",
        Test3::B(_) => "B(Empty)",
        Test3::C => "C",
        Test3::D => "D",
    };

    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;

    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: [[unreachable]], 1: [[unreachable]], 2: {{bb.*}}, 3: {{bb.*}}, otherwise: [[unreachable]]];
    match plop.test3 {
        Test3::A(_) => "A(Empty)",
        Test3::B(_) => "B(Empty)",
        Test3::C => "C",
        Test3::D => "D",
    };
}

fn main() {
    simple();
    custom_discriminant();
    otherwise_t1();
    otherwise_t2();
    otherwise_t3();
    otherwise_t4_unreachable_default();
    otherwise_t4_unreachable_default_2();
    otherwise_t4();
    otherwise_t5_unreachable_default::<i32>();
    byref();
}
