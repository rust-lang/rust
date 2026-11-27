trait Tr {
    type Assoc;
}

struct W<'a>(&'a ());

impl Tr for W<'_> {
    type Assoc = ();
}

// The normal way of capturing `'a`...
impl<'a> W<'a> {
    fn good1() -> impl Into<<W<'a> as Tr>::Assoc> + use<'a> {}
}

// This ensures that we don't error when we capture the *parent* copy of `'a`,
// since the opaque captures that rather than the duplicated `'a` lifetime
// synthesized from mentioning `'a` directly in the bounds.
impl<'a> W<'a> {
    fn good2() -> impl Into<<Self as Tr>::Assoc> + use<'a> {}
}

// The normal way of capturing `'a`... but not mentioned in the bounds.
impl<'a> W<'a> {
    fn bad1() -> impl Into<<W<'a> as Tr>::Assoc> + use<> {}
    //~^ ERROR `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list
}

// But also make sure that we error here...
impl<'a> W<'a> {
    fn bad2() -> impl Into<<Self as Tr>::Assoc> + use<> {}
    //~^ ERROR `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list
}

fn main() {}
