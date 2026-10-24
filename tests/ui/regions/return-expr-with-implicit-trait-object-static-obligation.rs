// Detect. #78547.

mod a {
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
    pub struct Thing(Box<dyn Iterator<Item = char>>);

    impl Thing {
        pub fn new(string: &str) -> Self {
            Thing(Box::new(string.chars())) //~ ERROR: lifetime may not live long enough
        }
    }
}

mod d {
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

fn main() {}
