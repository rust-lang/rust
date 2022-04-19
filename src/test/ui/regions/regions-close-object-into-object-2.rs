// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait A<T> { }

struct B<'a, T:'a>(&'a (dyn A<T> + 'a));

trait X { }
impl<'a, T> X for B<'a, T> {}

fn g<'a, T: 'static>(v: Box<dyn A<T> + 'a>) -> Box<dyn X + 'static> {
    Box::new(B(&*v)) as Box<dyn X>
    //[base]~^ ERROR E0759
    //[nll]~^^ ERROR lifetime may not live long enough
    //[nll]~| ERROR cannot return value referencing local data `*v` [E0515]
}

fn main() { }
