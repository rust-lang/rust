// Checks that manual WF bounds are not use for implied bounds.
//
// With the current implementation for WF check, this can easily cause
// unsoundness, as wf does not emit any obligations containing
// placeholders or bound variables.
struct MyStruct<T, U>(T, U);

trait Foo<'a, 'b> {}

// IF THIS TEST STOPS EMITTING ERRORS, PLEASE NOTIFY T-TYPES TO CHECK WHETHER THE CHANGE IS SOUND.
impl<'a, 'b> Foo<'a, 'b> for () //~ ERROR cannot infer an appropriate lifetime
where
    &'a &'b ():,
    //~^ ERROR in type `&'a &'b ()`, reference has a longer lifetime than the data it references
{}

fn main() {}
