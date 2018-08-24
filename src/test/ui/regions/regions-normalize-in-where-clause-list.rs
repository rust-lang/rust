// Test that we are able to normalize in the list of where-clauses,
// even if `'a: 'b` is required.

trait Project<'a, 'b> {
    type Item;
}

impl<'a, 'b> Project<'a, 'b> for ()
    where 'a: 'b
{
    type Item = ();
}

// No error here, we have 'a: 'b. We used to report an error here
// though, see https://github.com/rust-lang/rust/issues/45937.
fn foo<'a: 'b, 'b>()
    where <() as Project<'a, 'b>>::Item : Eq
{
}

// Here we get an error: we need `'a: 'b`.
fn bar<'a, 'b>() //~ ERROR cannot infer
    where <() as Project<'a, 'b>>::Item : Eq
{
}

fn main() { }
