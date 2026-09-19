// rustfmt-use_field_init_shorthand: true

foo!(Foo { a: a });

fn main() {
    foo!(Foo { a: a });
    foo![Foo { a: a }];
    foo! { Foo { a: a } }

    seq!(Task { a });
    seq!(Task { a: b });
    seq!(Task { a: 1 });
    seq!(Task { a: a, ..base });
    seq!(Task { #[attr] a: a });
    seq!(Task { a: Inner { b: b } });
    seq!(Task { a: a }.build());
    let _ = Foo { a: seq!(Task { b: b }) };

    // `vec!` is formatted as an array, so the shorthand still applies.
    vec![Foo { a: a }];
    vec![vec![Foo { a: a }]];
    foo!(vec![Foo { a: a }]);

    // std/core prelude macros, where the shorthand would be safe.
    assert_eq!(Foo { a: a }, x);
    debug_assert_eq!(Foo { a: a }, x);
    matches!(v, Foo { a: a });
    assert_matches!(v, Foo { a: a });
    assert!(matches!(v, Foo { a: a }));

    // A shorthand already written by hand is kept.
    assert_eq!(Foo { a }, x);
    debug_assert_eq!(Foo { a }, x);
    matches!(v, Foo { a });
    assert_matches!(v, Foo { a });
    assert!(matches!(v, Foo { a }));

    let _ = Foo { a: a };
    let _ = Foo { a: Inner { b: b } };
}
