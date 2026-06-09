// Demonstrates a mostly-theoretical inference guidance now that we turn the where
// clause on `Trait` into an item bound, given that we prefer item bounds somewhat
// greedily in trait selection.

trait Bound<T> {}
impl<T, U> Bound<T> for U {}

trait Trait
where
    <<Self as Trait>::Assoc as Other>::Assoc: Bound<u32>,
{
    type Assoc: Other;
}

trait Other {
    type Assoc;
}

fn impls_trait<T: Bound<U>, U>() -> Vec<U> { vec![] }

fn foo<T: Trait>() {
    let mut vec_u = impls_trait::<<<T as Trait>::Assoc as Other>::Assoc, _>();
    vec_u.sort();
    drop::<Vec<u8>>(vec_u);
    //~^ ERROR mismatched types
}

fn main() {}
