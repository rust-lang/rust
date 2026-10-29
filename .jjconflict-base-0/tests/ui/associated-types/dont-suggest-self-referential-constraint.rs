// Regression test for #112104.
//
// Don't suggest `Item = &<I as Iterator>::Item` when
// the expected type wraps the found projection.

fn option_of_ref_assoc<I: Iterator>(iter: &mut I) {
    let _: Option<&I::Item> = iter.next();
    //~^ ERROR mismatched types
}

// Valid constraint suggestions should still fire.
trait Foo {
    type Assoc;
}

fn assoc_to_concrete<T: Foo>(x: T::Assoc) -> u32 {
    x //~ ERROR mismatched types
}

fn main() {}
