//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@[old] check-pass

// cc #119820

trait Trait<T, U> {}

// using this impl results in a higher-ranked region error.
impl<'a> Trait<&'a str, &'a str> for () {}

impl<'a> Trait<&'a str, String> for () {}

fn impls_trait<T: for<'a> Trait<&'a str, U>, U>() {}

fn main() {
    impls_trait::<(), _>();
    //[next]~^ ERROR type annotations needed
}
