pub trait Foo {
    type Associated;
}

pub struct X;
pub struct Y;


impl Foo for X {
    type Associated = ();
}

impl Foo for Y {
    type Associated = ();
}

impl X {
    pub fn returns_sized<'a>(&'a self) -> impl Foo<Associated=()> + 'a {
        X
    }
}

impl Y {
    pub fn returns_unsized<'a>(&'a self) -> Box<impl ?Sized + Foo<Associated=()> + 'a> {
        Box::new(X)
    }
}
