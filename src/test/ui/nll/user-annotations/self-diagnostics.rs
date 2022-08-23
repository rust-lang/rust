// Some subtle cases where `Self` imposes unnecessary lifetime constraints.
// Make sure we detect these cases and suggest a proper fix.

// check-fail

struct S<'a>(&'a str);

impl S<'_> {
    const CLOSURE_ARGS: () = {
        |s: &str| {
            Self(s);
            //~^ ERROR lifetime may not live long enough
        };
    };

    fn closure_body() {
        let closure = |s| {
            Self(s);
        };
        closure(&String::new());
        //~^ ERROR temporary value dropped while borrowed
    }
}

impl<'x> S<'x> {
    fn static_method(_: &'x str) {}

    // FIXME suggesting replacing `Self` is better than the current one.
    fn test3(s: &str) {
        let _ = || {
            Self::static_method(s);
            //~^ ERROR explicit lifetime required in the type of `s`
        };
    }

    fn test_named_lt<'y>(s: &'y str) {
        let _ = || {
            Self::static_method(s);
            //~^ ERROR lifetime may not live long enough
        };
    }
}

impl<'x> S<'x> {
    fn test7(s: String) {
        s.split('/')
            //~^ ERROR `s` does not live long enough
            .map(|s| Self(s))
            .collect::<Vec<_>>();
    }

    // FIXME there is no suggestion here because we don't have a
    // correct span for closure argument annotations.
    fn test8(s: String) {
        s.split('/')
            //~^ ERROR `s` does not live long enough
            .map(|s| S(s))
            .map(|_: Self| {})
            .collect::<Vec<_>>();
    }
}

impl<'x> S<'x> {
    fn eq<T>(self, _: T) {}

    // test annotations on method call.
    fn method_call<'c>(self, s: &'c str) {
        self.eq::<Self>(S(s)); // remove Self
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
