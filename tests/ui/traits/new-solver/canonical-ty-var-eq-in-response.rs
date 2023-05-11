// check-pass
// compile-flags: -Ztrait-solver=next

trait Mirror {
    type Item;
}

struct Wrapper<T>(T);
impl<T> Mirror for Wrapper<T> {
    type Item = T;
}

fn mirror<T>()
where
    Wrapper<T>: Mirror<Item = i32>,
{
}

fn main() {
    mirror::<_ /* ?0 */>();

    // Solving `<Wrapper<?0> as Mirror>::Item = i32`

    // First, we replace the term with a fresh infer var:
    // `<Wrapper<?0> as Mirror>::Item = ?1`

    // We select the impl candidate on line #6, which leads us to learn that
    // `?0 == ?1`.

    // That should be reflected in our canonical response, which should have
    // `^0 = ^0, ^1 = ^0`
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !! We used to return a totally unconstrained response here :< !!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // Then, during the "equate term" part of the projection solving, we
    // instantiate the response from the unconstrained projection predicate,
    // and equate `?0 == i32`.
}
