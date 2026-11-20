//@ known-bug: #151069
trait Trait {
    type Assoc2;
}
struct Bar;
impl Trait for Bar
where
    <Bar as Trait>::Assoc2: Trait,
{
    type Assoc2 = ();
}
struct Foo {
    field: <Bar as Trait>::Assoc2,
}
static FOO2: &Foo = 0;
fn main() {}
