#![warn(clippy::needless_lifetimes, clippy::elidable_lifetime_names)]

type Ref<'r> = &'r u8;

// No error; same lifetime on two params.
fn lifetime_param_1<'a>(_x: Ref<'a>, _y: &'a u8) {}

//~v ERROR: could be elided: 'a, 'b
fn lifetime_param_2<'a, 'b>(_x: Ref<'a>, _y: &'b u8) {}

// No error; bounded lifetime.
fn lifetime_param_3<'a, 'b: 'a>(_x: Ref<'a>, _y: &'b u8) {}

// No error; bounded lifetime.
fn lifetime_param_4<'a, 'b>(_x: Ref<'a>, _y: &'b u8)
where
    'b: 'a,
{
}

struct Lt<'a, I: 'static> {
    x: &'a I,
}

// No error; fn bound references `'a`.
fn fn_bound<'a, F, I>(_m: Lt<'a, I>, _f: F) -> Lt<'a, I>
where
    F: Fn(Lt<'a, I>) -> Lt<'a, I>,
{
    unreachable!()
}

//~v ERROR: could be elided: 'a
fn fn_bound_2<'a, F, I>(_m: Lt<'a, I>, _f: F) -> Lt<'a, I>
where
    for<'x> F: Fn(Lt<'x, I>) -> Lt<'x, I>,
{
    unreachable!()
}

struct Foo<'a>(&'a u8);

//~v ERROR: could be elided: 'a
fn struct_with_lt<'a>(_foo: Foo<'a>) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (named on the reference, anonymous on `Foo`).
fn struct_with_lt2<'a>(_foo: &'a Foo) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (anonymous on the reference, named on `Foo`).
fn struct_with_lt3<'a>(_foo: &Foo<'a>) -> &'a str {
    unimplemented!()
}

//~v ERROR: could be elided: 'b
fn struct_with_lt4a<'a, 'b>(_foo: &'a Foo<'b>) -> &'a str {
    unimplemented!()
}

type FooAlias<'a> = Foo<'a>;

//~v ERROR: could be elided: 'a
fn alias_with_lt<'a>(_foo: FooAlias<'a>) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (named on the reference, anonymous on `FooAlias`).
fn alias_with_lt2<'a>(_foo: &'a FooAlias) -> &'a str {
    unimplemented!()
}

// No warning; two input lifetimes (anonymous on the reference, named on `FooAlias`).
fn alias_with_lt3<'a>(_foo: &FooAlias<'a>) -> &'a str {
    unimplemented!()
}

//~v ERROR: could be elided: 'b
fn alias_with_lt4a<'a, 'b>(_foo: &'a FooAlias<'b>) -> &'a str {
    unimplemented!()
}

// Issue #3284: give hint regarding lifetime in return type.
struct Cow<'a> {
    x: &'a str,
}

//~v ERROR: could be elided: 'a
fn out_return_type_lts<'a>(e: &'a str) -> Cow<'a> {
    unimplemented!()
}

mod issue2944 {
    trait Foo {}
    struct Bar;
    struct Baz<'a> {
        bar: &'a Bar,
    }

    //~v ERROR: could be elided: 'a
    impl<'a> Foo for Baz<'a> {}
    impl Bar {
        //~v ERROR: could be elided: 'a
        fn baz<'a>(&'a self) -> impl Foo + 'a {
            Baz { bar: self }
        }
    }
}

mod issue13923 {
    struct Py<'py> {
        data: &'py str,
    }

    enum Content<'t, 'py> {
        Py(Py<'py>),
        T1(&'t str),
        T2(&'t str),
    }

    enum ContentString<'t> {
        T1(&'t str),
        T2(&'t str),
    }

    impl<'t, 'py> ContentString<'t> {
        // `'py` cannot be elided
        fn map_content1(self, f: impl FnOnce(&'t str) -> &'t str) -> Content<'t, 'py> {
            match self {
                Self::T1(content) => Content::T1(f(content)),
                Self::T2(content) => Content::T2(f(content)),
            }
        }
    }

    //~v ERROR: could be elided: 'py
    impl<'t, 'py> ContentString<'t> {
        // `'py` can be elided because of `&self`
        fn map_content2(&self, f: impl FnOnce(&'t str) -> &'t str) -> Content<'t, 'py> {
            match self {
                Self::T1(content) => Content::T1(f(content)),
                Self::T2(content) => Content::T2(f(content)),
            }
        }
    }

    //~v ERROR: could be elided: 'py
    impl<'t, 'py> ContentString<'t> {
        // `'py` can be elided because of `&'_ self`
        fn map_content3(&'_ self, f: impl FnOnce(&'t str) -> &'t str) -> Content<'t, 'py> {
            match self {
                Self::T1(content) => Content::T1(f(content)),
                Self::T2(content) => Content::T2(f(content)),
            }
        }
    }

    impl<'t, 'py> ContentString<'t> {
        // `'py` should not be elided as the default lifetime, even if working, could be named as `'t`
        fn map_content4(self, f: impl FnOnce(&'t str) -> &'t str, o: &'t str) -> Content<'t, 'py> {
            match self {
                Self::T1(content) => Content::T1(f(content)),
                Self::T2(_) => Content::T2(o),
            }
        }
    }

    //~v ERROR: could be elided: 'py
    impl<'t, 'py> ContentString<'t> {
        // `'py` can be elided because of `&Self`
        fn map_content5(
            self: std::pin::Pin<&Self>,
            f: impl FnOnce(&'t str) -> &'t str,
            o: &'t str,
        ) -> Content<'t, 'py> {
            match *self {
                Self::T1(content) => Content::T1(f(content)),
                Self::T2(_) => Content::T2(o),
            }
        }
    }

    struct Cx<'a, 'b> {
        a: &'a u32,
        b: &'b u32,
    }

    // `'c` cannot be elided because we have several input lifetimes
    fn one_explicit<'b>(x: Cx<'_, 'b>) -> &'b u32 {
        x.b
    }
}
