// Test comments on functions are preserved.

// Comment on foo.
fn foo<F, G>(
    a: aaaaaaaaaaaaa, // A comment
    b: bbbbbbbbbbbbb, // a second comment
    c: ccccccccccccc,
    // Newline comment
    d: ddddddddddddd,
    //  A multi line comment
    // between args.
    e: eeeeeeeeeeeee, /* comment before paren */
) -> bar
where
    F: Foo, // COmment after where-clause
    G: Goo, // final comment
{
}

fn bar<F /* comment on F */, G /* comment on G */>() {}

fn baz() -> Baz /* Comment after return type */ {}

fn some_fn<T>()
where
    T: Eq, // some comment
{
}

fn issue458<F>(a: &str, f: F)
// comment1
where
    // comment2
    F: FnOnce(&str) -> bool,
{
    f(a);
    ()
}
