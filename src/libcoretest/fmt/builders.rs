mod debug_struct {
    use std::fmt;

    #[test]
    fn test_empty() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .finish()
            }
        }

        assert_eq!("Foo { bar: true }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    bar: true
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .field("baz", &format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        assert_eq!("Foo { bar: true, baz: 10/20 }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    bar: true,
    baz: 10/20
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_struct("Foo")
                    .field("bar", &true)
                    .field("baz", &format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
        baz: 10/20
    },
    hello: \"world\"
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_tuple("Foo")
                    .field(&true)
                    .finish()
            }
        }

        assert_eq!("Foo(true)", format!("{:?}", Foo));
        assert_eq!(
"Foo(
    true
)",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_tuple("Foo")
                    .field(&true)
                    .field(&format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        assert_eq!("Foo(true, 10/20)", format!("{:?}", Foo));
        assert_eq!(
"Foo(
    true,
    10/20
)",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_tuple("Foo")
                    .field(&true)
                    .field(&format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
        10/20
    ),
    \"world\"
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map("Foo").finish()
            }
        }

        assert_eq!("Foo {}", format!("{:?}", Foo));
        assert_eq!("Foo {}", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map("Foo")
                    .entry(&"bar", &true)
                    .finish()
            }
        }

        assert_eq!("Foo { \"bar\": true }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    \"bar\": true
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map("Foo")
                    .entry(&"bar", &true)
                    .entry(&10i32, &format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        assert_eq!("Foo { \"bar\": true, 10: 10/20 }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    \"bar\": true,
    10: 10/20
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map("Foo")
                    .entry(&"bar", &true)
                    .entry(&10i32, &format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map("Bar")
                    .entry(&"foo", &Foo)
                    .entry(&Foo, &"world")
                    .finish()
            }
        }

        assert_eq!("Bar { \"foo\": Foo { \"bar\": true, 10: 10/20 }, \
                    Foo { \"bar\": true, 10: 10/20 }: \"world\" }",
                   format!("{:?}", Bar));
        assert_eq!(
"Bar {
    \"foo\": Foo {
        \"bar\": true,
        10: 10/20
    },
    Foo {
        \"bar\": true,
        10: 10/20
    }: \"world\"
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set("Foo").finish()
            }
        }

        assert_eq!("Foo {}", format!("{:?}", Foo));
        assert_eq!("Foo {}", format!("{:#?}", Foo));
    }

    #[test]
    fn test_single() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set("Foo")
                    .entry(&true)
                    .finish()
            }
        }

        assert_eq!("Foo { true }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    true
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set("Foo")
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        assert_eq!("Foo { true, 10/20 }", format!("{:?}", Foo));
        assert_eq!(
"Foo {
    true,
    10/20
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set("Foo")
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10i32, 20i32))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set("Bar")
                    .entry(&Foo)
                    .entry(&"world")
                    .finish()
            }
        }

        assert_eq!("Bar { Foo { true, 10/20 }, \"world\" }",
                   format!("{:?}", Bar));
        assert_eq!(
"Bar {
    Foo {
        true,
        10/20
    },
    \"world\"
}",
                   format!("{:#?}", Bar));
    }
}
