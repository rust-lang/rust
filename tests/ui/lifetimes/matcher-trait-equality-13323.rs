//! Regression test for https://github.com/rust-lang/rust/issues/13323

//@ run-pass

struct StrWrap {
    s: String
}

impl StrWrap {
    fn new(s: &str) -> StrWrap {
        StrWrap { s: s.to_string() }
    }

    fn get_s<'a>(&'a self) -> &'a str {
        &self.s
    }
}

struct MyStruct {
    s: StrWrap
}

impl MyStruct {
    fn new(s: &str) -> MyStruct {
        MyStruct { s: StrWrap::new(s) }
    }

    fn get_str_wrap<'a>(&'a self) -> &'a StrWrap {
        &self.s
    }
}

trait Matcher<T> {
    fn matches(&self, actual: T) -> bool;
}

fn assert_that<T, U: Matcher<T>>(actual: T, matcher: &U) {
    assert!(matcher.matches(actual));
}

struct EqualTo<T> {
    expected: T
}

impl<T: Eq> Matcher<T> for EqualTo<T> {
    fn matches(&self, actual: T) -> bool {
        self.expected.eq(&actual)
    }
}

fn equal_to<T: Eq>(expected: T) -> Box<EqualTo<T>> {
    Box::new(EqualTo { expected: expected })
}

pub fn main() {
    let my_struct = MyStruct::new("zomg");
    let s = my_struct.get_str_wrap();

    assert_that(s.get_s(), &*equal_to("zomg"));
}
