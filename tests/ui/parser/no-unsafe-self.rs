trait A {
    fn foo(*mut self); //~ ERROR cannot pass `self` by raw pointer
    fn baz(*const self); //~ ERROR cannot pass `self` by raw pointer
    fn bar(*self); //~ ERROR cannot pass `self` by raw pointer
}

struct X;
impl A for X {
    fn foo(*mut self) { } //~ ERROR cannot pass `self` by raw pointer
    fn baz(*const self) { } //~ ERROR cannot pass `self` by raw pointer
    fn bar(*self) { } //~ ERROR cannot pass `self` by raw pointer
}

fn main() { }
