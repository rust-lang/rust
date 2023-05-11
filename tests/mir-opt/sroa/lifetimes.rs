// unit-test: ScalarReplacementOfAggregates
// compile-flags: -Cpanic=abort
// no-prefer-dynamic

trait Err {
    type Err;
}

struct Foo<T: Err> {
    // Check that the `'static` lifetime is erased when creating the local for `x`,
    // even if we fail to normalize the type.
    x: Result<Box<dyn std::fmt::Display + 'static>, <T as Err>::Err>,
    y: u32,
}

// EMIT_MIR lifetimes.foo.ScalarReplacementOfAggregates.diff
fn foo<T: Err>() {
    let foo: Foo<T> = Foo {
        x: Ok(Box::new(5_u32)),
        y: 7_u32,
    };

    let x = foo.x;
    let y = foo.y;

    if let Ok(x) = x {
        eprintln!("{x} {y}");
    }
}

impl Err for () {
    type Err = ();
}

fn main() {
    foo::<()>()
}
