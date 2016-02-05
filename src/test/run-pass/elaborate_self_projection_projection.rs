trait D {}

trait C {
    type D;
}

trait B {
    type C: Default;
    fn c(&self) -> Self::C { Default::default() }
}

trait A
    where <Self::B as B>::C: C, <<Self::B as B>::C as C>::D: D
{
    type B: B + Default;
    fn b(&self) -> Self::B { Default::default() }
}


trait QQ: A {}


impl D for () {}
impl C for i8 { type D = (); }
impl B for i16 { type C = i8; }
impl A for i32 { type B = i16; }
impl QQ for i32 {}

fn accept_a<T: A>(a: T) {
    let _ = a.b().c();
}
fn accept_qq<T: QQ>(a: T) {
    let _ = a.b().c();
}

pub fn main() {
    accept_a(0);
    accept_qq(0);
}
