// rustfmt-indent_style: Block
// rustfmt-combine_control_expr: false

// Combining openings and closings. See rust-lang/fmt-rfcs#61.

fn main() {
    // Call
    foo(bar(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ));

    // Mac
    foo(foo!(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ));

    // MethodCall
    foo(x.foo::<Bar, Baz>(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ));

    // Block
    foo!({
        foo();
        bar();
    });

    // Closure
    foo(|x| {
        let y = x + 1;
        y
    });

    // Match
    foo(match opt {
        Some(x) => x,
        None => y,
    });

    // Struct
    foo(Bar {
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    });

    // If
    foo!(
        if x {
            foo();
        } else {
            bar();
        }
    );

    // IfLet
    foo!(
        if let Some(..) = x {
            foo();
        } else {
            bar();
        }
    );

    // While
    foo!(
        while x {
            foo();
            bar();
        }
    );

    // WhileLet
    foo!(
        while let Some(..) = x {
            foo();
            bar();
        }
    );

    // ForLoop
    foo!(
        for x in y {
            foo();
            bar();
        }
    );

    // Loop
    foo!(
        loop {
            foo();
            bar();
        }
    );

    // Tuple
    foo((
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ));

    // AddrOf
    foo(&bar(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ));

    // Unary
    foo(!bar(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    ));

    // Try
    foo(bar(
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,
    )?);

    // Cast
    foo(Bar {
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
    } as i64);
}
