#![feature(rustc_attrs)]
#![rustc_variance_of_opaques]

trait Bar<'a> {
    type Assoc: From<()>;
}

fn foo<'a, T: Bar<'a>>() -> impl Into<T::Assoc> {
    //~^ ERROR ['a: o, T: o]
    // captures both T and 'a invariantly
    ()
}

fn foo2<'a, T: Bar<'a>>() -> impl Into<T::Assoc> + 'a {
    //~^ ERROR ['a: o, T: o, 'a: o]
    // captures both T and 'a invariantly, and also duplicates `'a`
    // i.e. the opaque looks like `impl Into<<T as Bar<'a>>::Assoc> + 'a_duplicated`
    ()
}

fn main() {}
