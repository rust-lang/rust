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
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .finish()
            }
        }

        assert_eq!("Foo { bar: true }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    bar: true,
}",
                   format!("{:#?}", Foo));
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
                   format!("{:#?}", Foo));
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
                fmt.debug_struct("Bar")
                    .field("foo", &Foo)
                    .field("hello", &"world")
                    .finish()
            }
        }

        assert_eq!("Bar { foo: Foo { bar: true, baz: 10/20 }, hello: \"world\" }",
                   format!("{:?}", Bar));
        assert_eq!(
"Bar {
    foo: Foo {
        bar: true,
        baz: 10/20,
    },
    hello: \"world\",
}",
                   format!("{:#?}", Bar));
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
                fmt.debug_tuple("Foo")
                    .field(&true)
                    .finish()
            }
        }

        assert_eq!("Foo(true)", format!("{:?}", Foo));
        assert_eq!(
"Foo(
    true,
)",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Foo")
                    .field(&true)
                    .field(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        assert_eq!("Foo(true, 10/20)", format!("{:?}", Foo));
        assert_eq!(
"Foo(
    true,
    10/20,
)",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Foo")
                    .field(&true)
                    .field(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple("Bar")
                    .field(&Foo)
                    .field(&"world")
                    .finish()
            }
        }

        assert_eq!("Bar(Foo(true, 10/20), \"world\")",
                   format!("{:?}", Bar));
        assert_eq!(
"Bar(
    Foo(
        true,
        10/20,
    ),
    \"world\",
)",
                   format!("{:#?}", Bar));
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
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map()
                    .entry(&"bar", &true)
                    .finish()
            }
        }

        assert_eq!("{\"bar\": true}", format!("{:?}", Foo));
        assert_eq!(
"{
    \"bar\": true,
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_map()
                    .entry(&"bar", &true)
                    .entry(&10, &format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        assert_eq!("{\"bar\": true, 10: 10/20}", format!("{:?}", Foo));
        assert_eq!(
"{
    \"bar\": true,
    10: 10/20,
}",
                   format!("{:#?}", Foo));
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
                fmt.debug_map()
                    .entry(&"foo", &Foo)
                    .entry(&Foo, &"world")
                    .finish()
            }
        }

        assert_eq!("{\"foo\": {\"bar\": true, 10: 10/20}, \
                    {\"bar\": true, 10: 10/20}: \"world\"}",
                   format!("{:?}", Bar));
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
                   format!("{:#?}", Bar));
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
                fmt.debug_set()
                    .entry(&true)
                    .finish()
            }
        }

        assert_eq!("{true}", format!("{:?}", Foo));
        assert_eq!(
"{
    true,
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set()
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        assert_eq!("{true, 10/20}", format!("{:?}", Foo));
        assert_eq!(
"{
    true,
    10/20,
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set()
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_set()
                    .entry(&Foo)
                    .entry(&"world")
                    .finish()
            }
        }

        assert_eq!("{{true, 10/20}, \"world\"}",
                   format!("{:?}", Bar));
        assert_eq!(
"{
    {
        true,
        10/20,
    },
    \"world\",
}",
                   format!("{:#?}", Bar));
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
                fmt.debug_list()
                    .entry(&true)
                    .finish()
            }
        }

        assert_eq!("[true]", format!("{:?}", Foo));
        assert_eq!(
"[
    true,
]",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list()
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        assert_eq!("[true, 10/20]", format!("{:?}", Foo));
        assert_eq!(
"[
    true,
    10/20,
]",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list()
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_list()
                    .entry(&Foo)
                    .entry(&"world")
                    .finish()
            }
        }

        assert_eq!("[[true, 10/20], \"world\"]",
                   format!("{:?}", Bar));
        assert_eq!(
"[
    [
        true,
        10/20,
    ],
    \"world\",
]",
                   format!("{:#?}", Bar));
    }
}

#[test]
fn test_formatting_parameters_are_forwarded() {
    use std::collections::{BTreeMap, BTreeSet};
    #[derive(Debug)]
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
    assert_eq!(format!("{:#03?}", struct_), "
Foo {
    bar: 1024,
    baz: 007,
}
    ".trim());
    assert_eq!(format!("{:#03?}", tuple), "
(
    1024,
    007,
)
    ".trim());
    assert_eq!(format!("{:#03?}", list), "
[
    1024,
    007,
]
    ".trim());
    assert_eq!(format!("{:#03?}", map), r#"
{
    "bar": 1024,
    "baz": 007,
}
    "#.trim());
    assert_eq!(format!("{:#03?}", set), "
{
    007,
    1024,
}
    ".trim());
}
