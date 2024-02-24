//@ unit-test: UninhabitedEnumBranching
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

struct Plop {
    xx: u32,
    test3: Test3,
}

// EMIT_MIR uninhabited_enum_branching.simple.UninhabitedEnumBranching.diff
fn simple() {
    // CHECK-LABEL: fn simple(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: [[unreachable:bb.*]], 1: [[unreachable]], 2: bb2, otherwise: [[unreachable]]];
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    match Test1::C {
        Test1::A(_) => "A(Empty)",
        Test1::B(_) => "B(Empty)",
        Test1::C => "C",
    };
}

// EMIT_MIR uninhabited_enum_branching.custom_discriminant.UninhabitedEnumBranching.diff
fn custom_discriminant() {
    // CHECK-LABEL: fn custom_discriminant(
    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [4: bb3, 5: bb2, otherwise: bb5];
    // CHECK: bb5: {
    // CHECK-NEXT: unreachable;
    match Test2::D {
        Test2::D => "D",
        Test2::E => "E",
    };
}

// EMIT_MIR uninhabited_enum_branching.byref.UninhabitedEnumBranching.diff
fn byref() {
    // CHECK-LABEL: fn byref(
    let plop = Plop { xx: 51, test3: Test3::C };

    // CHECK: [[ref_discr:_.*]] = discriminant((*
    // CHECK: switchInt(move [[ref_discr]]) -> [0: [[unreachable:bb.*]], 1: [[unreachable]], 2: bb5, 3: bb2, otherwise: [[unreachable]]];
    match &plop.test3 {
        Test3::A(_) => "A(Empty)",
        Test3::B(_) => "B(Empty)",
        Test3::C => "C",
        Test3::D => "D",
    };

    // CHECK: [[discr:_.*]] = discriminant(
    // CHECK: switchInt(move [[discr]]) -> [0: [[unreachable]], 1: [[unreachable]], 2: bb10, 3: bb7, otherwise: [[unreachable]]];
    match plop.test3 {
        Test3::A(_) => "A(Empty)",
        Test3::B(_) => "B(Empty)",
        Test3::C => "C",
        Test3::D => "D",
    };

    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
}

fn main() {
    simple();
    custom_discriminant();
    byref();
}
