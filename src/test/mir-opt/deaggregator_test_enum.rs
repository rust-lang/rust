enum Baz {
    Empty,
    Foo { x: usize },
}

// EMIT_MIR deaggregator_test_enum.bar.Deaggregator.diff
fn bar(a: usize) -> Baz {
    Baz::Foo { x: a }
}

fn main() {
    let x = bar(10);
    match x {
        Baz::Empty => println!("empty"),
        Baz::Foo { x } => println!("{}", x),
    };
}
