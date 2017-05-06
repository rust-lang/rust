// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
                    .field("baz", &format_args!("{}/{}", 10, 20))
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
                    .field("baz", &format_args!("{}/{}", 10, 20))
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
                    .field(&format_args!("{}/{}", 10, 20))
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
                    .field(&format_args!("{}/{}", 10, 20))
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map()
                    .entry(&"bar", &true)
                    .finish()
            }
        }

        assert_eq!("{\"bar\": true}", format!("{:?}", Foo));
        assert_eq!(
"{
    \"bar\": true
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    10: 10/20
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_map()
                    .entry(&"bar", &true)
                    .entry(&10, &format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
        10: 10/20
    },
    {
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set()
                    .entry(&true)
                    .finish()
            }
        }

        assert_eq!("{true}", format!("{:?}", Foo));
        assert_eq!(
"{
    true
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    10/20
}",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_set()
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
        10/20
    },
    \"world\"
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_list()
                    .entry(&true)
                    .finish()
            }
        }

        assert_eq!("[true]", format!("{:?}", Foo));
        assert_eq!(
"[
    true
]",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_multiple() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
    10/20
]",
                   format!("{:#?}", Foo));
    }

    #[test]
    fn test_nested() {
        struct Foo;

        impl fmt::Debug for Foo {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_list()
                    .entry(&true)
                    .entry(&format_args!("{}/{}", 10, 20))
                    .finish()
            }
        }

        struct Bar;

        impl fmt::Debug for Bar {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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
        10/20
    ],
    \"world\"
]",
                   format!("{:#?}", Bar));
    }
}
