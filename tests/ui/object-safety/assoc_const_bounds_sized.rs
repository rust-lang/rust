trait Foo {
    const BAR: bool
        where //~ ERROR: expected one of `!`, `(`, `+`, `::`, `;`, `<`, or `=`, found keyword `where`
            Self: Sized;
}

fn foo(_: &dyn Foo) {}

fn main() {}
