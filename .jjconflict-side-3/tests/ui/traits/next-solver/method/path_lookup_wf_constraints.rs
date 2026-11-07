//@ compile-flags: -Znext-solver
//@ check-pass

// A regression test for trait-system-refactor-initiative#161

trait Constrain<T> {
    type Assoc;
}
impl<T> Constrain<T> for () {
    type Assoc = ();
}
struct Foo<T, U = <() as Constrain<T>>::Assoc>(T, U);

impl<T: Copy> Foo<T> {
    fn foo() {}
}
struct B;
impl Foo<B> {
    fn foo() {}
}

type Alias<T> = Foo<T>;
fn via_guidance<T: Copy>()
where
    (): Constrain<T>,
{
    // Method selection on `Foo<?t, <() as Constrain<?t>>::Assoc>` is ambiguous.
    // only by unnecessarily constraining `?t` to `T` when proving `(): Constrain<?t>`
    // are we able to select the first impl.
    //
    // This happens in the old solver when normalizing `Alias<?t>`. The new solver doesn't try
    // to eagerly normalize `<() as Constrain<?t>>::Assoc` so we instead always prove that the
    // self type is well-formed before method lookup.
    Alias::foo();
}

fn main() {
    via_guidance::<()>();
}
