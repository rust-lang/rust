trait Bar {}

trait Foo {
    type Assoc: Bar;
}

impl Foo for () {
    type Assoc = bool; //~ ERROR trait `Bar` is not implemented for `bool`
}

trait Baz
where
    Self::Assoc: Bar,
{
    type Assoc;
}

impl Baz for () {
    type Assoc = bool; //~ ERROR trait `Bar` is not implemented for `bool`
}

trait Bat
where
    <Self as Bat>::Assoc: Bar,
{
    type Assoc;
}

impl Bat for () {
    type Assoc = bool; //~ ERROR trait `Bar` is not implemented for `bool`
}

fn main() {}
