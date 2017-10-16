// docs example
struct Foo(i32);
impl From<String> for Foo {
    fn from(s: String) -> Self {
        Foo(s.parse().unwrap())
    }
}


struct Valid(Vec<u8>);

impl<'a> From<&'a str> for Valid {
    fn from(s: &'a str) -> Valid {
        Valid(s.to_owned().into_bytes())
    }
}
impl From<String> for Valid {
    fn from(s: String) -> Valid {
        Valid(s.into_bytes())
    }
}
impl From<usize> for Valid {
    fn from(i: usize) -> Valid {
        if i == 0 {
            panic!();
        }
        Valid(Vec::with_capacity(i))
    }
}


struct Invalid;

impl<'a> From<&'a str> for Invalid {
    fn from(s: &'a str) -> Invalid {
        if !s.is_empty() {
            panic!();
        }
        Invalid
    }
}

impl From<String> for Invalid {
    fn from(s: String) -> Invalid {
        if !s.is_empty() {
            panic!(42);
        } else if s.parse::<u32>().unwrap() != 42 {
            panic!("{:?}", s);
        }
        Invalid
    }
}

trait ProjStrTrait {
    type ProjString;
}
impl<T> ProjStrTrait for Box<T> {
    type ProjString = String;
}
impl<'a> From<&'a mut <Box<u32> as ProjStrTrait>::ProjString> for Invalid {
    fn from(s: &'a mut <Box<u32> as ProjStrTrait>::ProjString) -> Invalid {
        if s.parse::<u32>().ok().unwrap() != 42 {
            panic!("{:?}", s);
        }
        Invalid
    }
}

fn main() {}
