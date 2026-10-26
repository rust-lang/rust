// Detect. #78547.

mod a {
    // Originally reported case, where `dyn Trait` had implicit `'static` requirement.
    // We ensure we point at `dyn Iterator<Item = char>` and we should be suggesting to add a named
    // lifetime to `Thing`.
    pub struct Thing {
        chars: Box<dyn Iterator<Item = char>>,
    }

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing {
                chars: Box::new(string.chars()), //~ ERROR: lifetime may not live long enough
            }
        }
    }
}

mod b {
    // Similar to above, but the `dyn Trait` has an explicit named bound on the type, but an
    // implicit one in the `impl`.
    // We ensure we point at `impl Thing<'_>` as introducing the unmet lifetime in `-> Self`.
    pub struct Thing<'a> {
        chars: Box<dyn Iterator<Item = char> + 'a>,
    }

    impl Thing<'_> {
        pub fn new(string: &str) -> Self {
            Thing { //~ ERROR: lifetime may not live long enough
                chars: Box::new(string.chars()),
            }
        }
    }
}

mod c {
    // Ensure that we are handling tuple-structs.
    pub struct Thing(Box<dyn Iterator<Item = char>>);

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing(Box::new(string.chars())) //~ ERROR: lifetime may not live long enough
        }
    }
}

mod d {
    // Ensure that we are handling tuple enum variants.
    pub enum Thing {
        Foo(Box<dyn Iterator<Item = char>>),
    }

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing::Foo(Box::new(string.chars())) //~ ERROR: lifetime may not live long enough
        }
    }
}

mod e {
    // Ensure that we are handling struct enum variants.
    pub enum Thing {
        Foo {
            chars: Box<dyn Iterator<Item = char>>,
        }
    }

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing::Foo {
                chars: Box::new(string.chars()), //~ ERROR: lifetime may not live long enough
            }
        }
    }
}

mod f {
    pub struct Thing<'a> {
        chars: Box<dyn Iterator<Item = char>>,
        other: &'a (),
    }

    impl Thing<'_> {
        pub fn new(string: &str) -> Self {
            Thing {
                chars: Box::new(string.chars()), //~ ERROR: lifetime may not live long enough
                other: &(),
            }
        }
    }
}

mod g {
    pub struct Thing<'a> {
        chars: Box<dyn Iterator<Item = char>>,
        other: &'a (),
    }

    impl<'a> Thing<'a> {
        pub fn new(string: &str) -> Self {
            Thing {
                chars: Box::new(string.chars()), //~ ERROR: lifetime may not live long enough
                other: &(),
            }
        }
    }
}

mod h {
    // Multiple lifetime failures on the same expression. We currently only talk about the first
    // field's lifetime failure, and point at the first field in the definition.
    pub struct Thing {
        chars: Box<dyn Iterator<Item = char>>,
        other_chars: Box<dyn Iterator<Item = char>>,
    }

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing {
                chars: Box::new(string.chars()), //~ ERROR: lifetime may not live long enough
                other_chars: Box::new(string.chars()),
            }
        }
    }
}

mod i {
    // Explicit check for more than one `dyn Trait` introducing the `'static` lifetime
    pub struct Thing {
        chars: (Box<dyn Iterator<Item = char>>, Box<dyn Iterator<Item = char>>),
    }

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing {
                chars: (Box::new(string.chars()), Box::new(string.chars())), //~ ERROR: lifetime may not live long enough
            }
        }
    }
}

mod j {
    // Multiple returned values with the same problem. We currently ignore the second incident.
    pub struct Thing {
        chars: Box<dyn Iterator<Item = char>>,
    }

    impl Thing {
        pub fn new(string: &str) -> Self {
            if true {
                return Thing {
                    chars: Box::new(string.chars()), //~ ERROR: lifetime may not live long enough
                };
            }
            Thing {
                chars: Box::new(string.chars()),
            }
        }
    }
}

mod k {
    // Similar to `b`, but with a named lifetime in the `impl`. We flat out tell people to use the
    // exisitng named lifetime in `fn new(&str)`.
    pub struct Thing<'a> {
        chars: Box<dyn Iterator<Item = char> + 'a>,
    }

    impl<'a> Thing<'a> {
        pub fn new(string: &str) -> Self {
            Thing { //~ ERROR: explicit lifetime required in the type of `string`
                chars: Box::new(string.chars()),
            }
        }
    }
}


fn main() {}
