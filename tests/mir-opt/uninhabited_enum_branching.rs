enum Empty { }

// test matching an enum with uninhabited variants
enum Test1 {
    A(Empty),
    B(Empty),
    C
}

// test an enum where the discriminants don't match the variant indexes
// (the optimization should do nothing here)
enum Test2 {
    D = 4,
    E = 5,
}

// EMIT_MIR uninhabited_enum_branching.main.UninhabitedEnumBranching.diff
// EMIT_MIR uninhabited_enum_branching.main.SimplifyCfg-after-uninhabited-enum-branching.after.mir
fn main() {
    match Test1::C {
        Test1::A(_) => "A(Empty)",
        Test1::B(_) => "B(Empty)",
        Test1::C => "C",
    };

    match Test2::D {
        Test2::D => "D",
        Test2::E => "E",
    };
}
