// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true
// rustfmt-struct_lit_single_line: false

// Struct literal expressions.

fn main() {
    let x = Bar;

    // Comment
    let y = Foo {
        a: x,
    };

    Foo {
        a: foo(), // comment
        // comment
        b: bar(),
        ..something
    };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo {
        a: foo(),
        b: bar(),
    };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo {
        a: foo(),
        b: bar(),
    };

    Foooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo {
        // Comment
        a: foo(), // Comment
        // Comment
        b: bar(), // Comment
    };

    Foo {
        a: Bar,
        b: foo(),
    };

    Quux {
        x: if cond {
            bar();
        },
        y: baz(),
    };

    A {
        // Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit
        // amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit.
        // Donec et mollis dolor.
        first: item(),
        // Praesent et diam eget libero egestas mattis sit amet vitae augue.
        // Nam tincidunt congue enim, ut porta lorem lacinia consectetur.
        second: Item,
    };

    Some(Data::MethodCallData(MethodCallData {
        span: sub_span.unwrap(),
        scope: self.enclosing_scope(id),
        ref_id: def_id,
        decl_id: Some(decl_id),
    }));

    Diagram {
        //                 o        This graph demonstrates how
        //                / \       significant whitespace is
        //               o   o      preserved.
        //              /|\   \
        //             o o o   o
        graph: G,
    }
}

fn matcher() {
    TagTerminatedByteMatcher {
        matcher: ByteMatcher {
            pattern: b"<HTML",
            mask: b"\xFF\xDF\xDF\xDF\xDF\xFF",
        },
    };
}

fn issue177() {
    struct Foo<T> {
        memb: T,
    }
    let foo = Foo::<i64> {
        memb: 10,
    };
}

fn issue201() {
    let s = S {
        a: 0,
        ..b
    };
}

fn issue201_2() {
    let s = S {
        a: S2 {
            ..c
        },
        ..b
    };
}

fn issue491() {
    Foo {
        guard: None,
        arm: 0, // Comment
    };
}
