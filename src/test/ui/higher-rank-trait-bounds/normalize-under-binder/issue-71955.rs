// ignore-compare-mode-nll
// revisions: migrate nll
// [nll]compile-flags: -Zborrowck=mir
// check-fail

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

#![feature(rustc_attrs)]

trait Parser<'s> {
    type Output;

    fn call(&self, input: &'s str) -> (&'s str, Self::Output);
}

impl<'s, F, T> Parser<'s> for F
where F: Fn(&'s str) -> (&'s str, T) {
    type Output = T;
    fn call(&self, input: &'s str) -> (&'s str, T) {
        self(input)
    }
}

fn foo<F1, F2>(
    f1: F1,
    base: &'static str,
    f2: F2
)
where
    F1: for<'a> Parser<'a>,
    F2: FnOnce(&<F1 as Parser>::Output) -> bool
{
    let s: String = base.to_owned();
    let str_ref = s.as_ref();
    let (remaining, produced) = f1.call(str_ref);
    assert!(f2(&produced));
    assert_eq!(remaining.len(), 0);
}

struct Wrapper<'a>(&'a str);

// Because nll currently succeeds and migrate doesn't
#[rustc_error]
fn main() {
    //[nll]~^ fatal
    fn bar<'a>(s: &'a str) -> (&'a str, &'a str) {
        (&s[..1], &s[..])
    }

    fn baz<'a>(s: &'a str) -> (&'a str, Wrapper<'a>) {
        (&s[..1], Wrapper(&s[..]))
    }

    foo(bar, "string", |s| s.len() == 5);
    //[migrate]~^ ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    foo(baz, "string", |s| s.0.len() == 5);
    //[migrate]~^ ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
    //[migrate]~| ERROR implementation of `Parser` is not general enough
}
