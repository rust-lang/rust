mod debug_struct {
    use std::fmt;

    #[test]
    fn test_empty() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo").finish()
            }
        }

        assert_eq!("Foo", format!("{:?}", Foo));
        assert_eq!("Foo", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo").field("bar", &true).finish()
            }
        }

        assert_eq!("Foo { bar: true }", format!("{:?}", Foo));
        assert_eq!(
            "Foo {
    bar: true,
}",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .field("baz", &format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        assert_eq!("Foo { bar: true, baz: 10/20 }", format!("{:?}", Foo));
        assert_eq!(
            "Foo {
    bar: true,
    baz: 10/20,
}",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .field("baz", &format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Bar").field("foo", &Foo).field("hello", &"world").finish()
            }
        }

        assert_eq!(
            "Bar { foo: Foo { bar: true, baz: 10/20 }, hello: \"world\" }",
            format!("{:?}", Bar)
        );
        assert_eq!(
            "Bar {
    foo: Foo {
        bar: true,
        baz: 10/20,
    },
    hello: \"world\",
}",
            format!("{:#?}", Bar)
        );
    }

    #[test]
    fn test_only_non_exhaustive() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo").finish_non_exhaustive()
            }
        }

        assert_eq!("Foo { .. }", format!("{:?}", Foo));
        assert_eq!("Foo { .. }", format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple_and_non_exhaustive() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .field("baz", &format_args!("{}/{}", 10, 20))
                    .finish_non_exhaustive()
            }
        }

        assert_eq!("Foo { bar: true, baz: 10/20, .. }", format!("{:?}", Foo));
        assert_eq!(
            "Foo {
    bar: true,
    baz: 10/20,
    ..
}",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_nested_non_exhaustive() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .field("baz", &format_args!("{}/{}", 10, 20))
                    .finish_non_exhaustive()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_struct("Bar")
                    .field("foo", &Foo)
                    .field("hello", &"world")
                    .finish_non_exhaustive()
            }
        }

        assert_eq!(
            "Bar { foo: Foo { bar: true, baz: 10/20, .. }, hello: \"world\", .. }",
            format!("{:?}", Bar)
        );
        assert_eq!(
            "Bar {
    foo: Foo {
        bar: true,
        baz: 10/20,
        ..
    },
    hello: \"world\",
    ..
}",
            format!("{:#?}", Bar)
        );
    }
}

mod debug_tuple {
    use std::fmt;

    #[test]
    fn test_empty() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Foo").finish()
            }
        }

        assert_eq!("Foo", format!("{:?}", Foo));
        assert_eq!("Foo", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Foo").field(&true).finish()
            }
        }

        assert_eq!("Foo(true)", format!("{:?}", Foo));
        assert_eq!(
            "Foo(
    true,
)",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Foo").field(&true).field(&format_args!("{}/{}", 10, 20)).finish()
            }
        }

        assert_eq!("Foo(true, 10/20)", format!("{:?}", Foo));
        assert_eq!(
            "Foo(
    true,
    10/20,
)",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Foo").field(&true).field(&format_args!("{}/{}", 10, 20)).finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Bar").field(&Foo).field(&"world").finish()
            }
        }

        assert_eq!("Bar(Foo(true, 10/20), \"world\")", format!("{:?}", Bar));
        assert_eq!(
            "Bar(
    Foo(
        true,
        10/20,
    ),
    \"world\",
)",
            format!("{:#?}", Bar)
        );
    }
}

mod debug_map {
    use std::fmt;

    #[test]
    fn test_empty() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().finish()
            }
        }

        assert_eq!("{}", format!("{:?}", Foo));
        assert_eq!("{}", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Entry;

        impl fmt::Debug for Entry {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().entry(&"bar", &true).finish()
            }
        }

        struct KeyValue;

        impl fmt::Debug for KeyValue {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().key(&"bar").value(&true).finish()
            }
        }

        assert_eq!(format!("{:?}", Entry), format!("{:?}", KeyValue));
        assert_eq!(format!("{:#?}", Entry), format!("{:#?}", KeyValue));

        assert_eq!("{\"bar\": true}", format!("{:?}", Entry));
        assert_eq!(
            "{
    \"bar\": true,
}",
            format!("{:#?}", Entry)
        );
    }

    #[test]
    fn test_multiple() {
        struct Entry;

        impl fmt::Debug for Entry {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map()
                    .entry(&"bar", &true)
                    .entry(&10, &format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct KeyValue;

        impl fmt::Debug for KeyValue {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map()
                    .key(&"bar")
                    .value(&true)
                    .key(&10)
                    .value(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        assert_eq!(format!("{:?}", Entry), format!("{:?}", KeyValue));
        assert_eq!(format!("{:#?}", Entry), format!("{:#?}", KeyValue));

        assert_eq!("{\"bar\": true, 10: 10/20}", format!("{:?}", Entry));
        assert_eq!(
            "{
    \"bar\": true,
    10: 10/20,
}",
            format!("{:#?}", Entry)
        );
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map()
                    .entry(&"bar", &true)
                    .entry(&10, &format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().entry(&"foo", &Foo).entry(&Foo, &"world").finish()
            }
        }

        assert_eq!(
            "{\"foo\": {\"bar\": true, 10: 10/20}, \
                    {\"bar\": true, 10: 10/20}: \"world\"}",
            format!("{:?}", Bar)
        );
        assert_eq!(
            "{
    \"foo\": {
        \"bar\": true,
        10: 10/20,
    },
    {
        \"bar\": true,
        10: 10/20,
    }: \"world\",
}",
            format!("{:#?}", Bar)
        );
    }

    #[test]
    fn test_entry_err() {
        // Ensure errors in a map entry don't trigger panics (#65231)
        use std::fmt::Write;

        struct ErrorFmt;

        impl fmt::Debug for ErrorFmt {
            fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
                Err(fmt::Error)
            }
        }

        struct KeyValue<K, V>(usize, K, V);

        impl<K, V> fmt::Debug for KeyValue<K, V>
        where
            K: fmt::Debug,
            V: fmt::Debug,
        {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut map = fmt.debug_map();

                for _ in 0..self.0 {
                    map.entry(&self.1, &self.2);
                }

                map.finish()
            }
        }

        let mut buf = String::new();

        assert!(write!(&mut buf, "{:?}", KeyValue(1, ErrorFmt, "bar")).is_err());
        assert!(write!(&mut buf, "{:?}", KeyValue(1, "foo", ErrorFmt)).is_err());

        assert!(write!(&mut buf, "{:?}", KeyValue(2, ErrorFmt, "bar")).is_err());
        assert!(write!(&mut buf, "{:?}", KeyValue(2, "foo", ErrorFmt)).is_err());
    }

    #[test]
    #[should_panic]
    fn test_invalid_key_when_entry_is_incomplete() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().key(&"bar").key(&"invalid").finish()
            }
        }

        format!("{:?}", Foo);
    }

    #[test]
    #[should_panic]
    fn test_invalid_finish_incomplete_entry() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().key(&"bar").finish()
            }
        }

        format!("{:?}", Foo);
    }

    #[test]
    #[should_panic]
    fn test_invalid_value_before_key() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map().value(&"invalid").key(&"bar").finish()
            }
        }

        format!("{:?}", Foo);
    }
}

mod debug_set {
    use std::fmt;

    #[test]
    fn test_empty() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set().finish()
            }
        }

        assert_eq!("{}", format!("{:?}", Foo));
        assert_eq!("{}", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set().entry(&true).finish()
            }
        }

        assert_eq!("{true}", format!("{:?}", Foo));
        assert_eq!(
            "{
    true,
}",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set().entry(&true).entry(&format_args!("{}/{}", 10, 20)).finish()
            }
        }

        assert_eq!("{true, 10/20}", format!("{:?}", Foo));
        assert_eq!(
            "{
    true,
    10/20,
}",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set().entry(&true).entry(&format_args!("{}/{}", 10, 20)).finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set().entry(&Foo).entry(&"world").finish()
            }
        }

        assert_eq!("{{true, 10/20}, \"world\"}", format!("{:?}", Bar));
        assert_eq!(
            "{
    {
        true,
        10/20,
    },
    \"world\",
}",
            format!("{:#?}", Bar)
        );
    }
}

mod debug_list {
    use std::fmt;

    #[test]
    fn test_empty() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().finish()
            }
        }

        assert_eq!("[]", format!("{:?}", Foo));
        assert_eq!("[]", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&true).finish()
            }
        }

        assert_eq!("[true]", format!("{:?}", Foo));
        assert_eq!(
            "[
    true,
]",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&true).entry(&format_args!("{}/{}", 10, 20)).finish()
            }
        }

        assert_eq!("[true, 10/20]", format!("{:?}", Foo));
        assert_eq!(
            "[
    true,
    10/20,
]",
            format!("{:#?}", Foo)
        );
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&true).entry(&format_args!("{}/{}", 10, 20)).finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list().entry(&Foo).entry(&"world").finish()
            }
        }

        assert_eq!("[[true, 10/20], \"world\"]", format!("{:?}", Bar));
        assert_eq!(
            "[
    [
        true,
        10/20,
    ],
    \"world\",
]",
            format!("{:#?}", Bar)
        );
    }
}

#[test]
fn test_formatting_parameters_are_forwarded() {
    use std::collections::{BTreeMap, BTreeSet};
    #[derive(Debug)]
    #[allow(dead_code)]
    struct Foo {
        bar: u32,
        baz: u32,
    }
    let struct_ = Foo { bar: 1024, baz: 7 };
    let tuple = (1024, 7);
    let list = [1024, 7];
    let mut map = BTreeMap::new();
    map.insert("bar", 1024);
    map.insert("baz", 7);
    let mut set = BTreeSet::new();
    set.insert(1024);
    set.insert(7);

    assert_eq!(format!("{:03?}", struct_), "Foo { bar: 1024, baz: 007 }");
    assert_eq!(format!("{:03?}", tuple), "(1024, 007)");
    assert_eq!(format!("{:03?}", list), "[1024, 007]");
    assert_eq!(format!("{:03?}", map), r#"{"bar": 1024, "baz": 007}"#);
    assert_eq!(format!("{:03?}", set), "{007, 1024}");
    assert_eq!(
        format!("{:#03?}", struct_),
        "
Foo {
    bar: 1024,
    baz: 007,
}
    "
        .trim()
    );
    assert_eq!(
        format!("{:#03?}", tuple),
        "
(
    1024,
    007,
)
    "
        .trim()
    );
    assert_eq!(
        format!("{:#03?}", list),
        "
[
    1024,
    007,
]
    "
        .trim()
    );
    assert_eq!(
        format!("{:#03?}", map),
        r#"
{
    "bar": 1024,
    "baz": 007,
}
    "#
        .trim()
    );
    assert_eq!(
        format!("{:#03?}", set),
        "
{
    007,
    1024,
}
    "
        .trim()
    );
}
