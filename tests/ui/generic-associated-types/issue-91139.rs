trait Foo<T> {
    type Type<'a>
    where
        T: 'a;
}

impl<T> Foo<T> for () {
    type Type<'a> = ()
    where
        T: 'a;
}

fn foo<T>() {
    let _: for<'a> fn(<() as Foo<T>>::Type<'a>, &'a T) = |_, _| ();
    //~^ ERROR `T` does not live long enough
    //~| ERROR `T` does not live long enough
    //
    // FIXME: This error is bogus, but it arises because we try to validate
    // that `<() as Foo<T>>::Type<'a>` is valid, which requires proving
    // that `T: 'a`. Since `'a` is higher-ranked, this becomes
    // `for<'a> T: 'a`, which is not true. Of course, the error is bogus
    // because there *ought* to be an implied bound stating that `'a` is
    // not any lifetime but specifically
    // "some `'a` such that `<() as Foo<T>>::Type<'a>" is valid".
}

pub fn main() {}
