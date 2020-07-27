// Test that deaggregate fires in more than one basic block

enum Foo {
    A(i32),
    B(i32),
}

// EMIT_MIR deaggregator_test_enum_2.test1.Deaggregator.diff
fn test1(x: bool, y: i32) -> Foo {
    if x {
        Foo::A(y)
    } else {
        Foo::B(y)
    }
}

fn main() {
    // Make sure the function actually gets instantiated.
    test1(false, 0);
}
