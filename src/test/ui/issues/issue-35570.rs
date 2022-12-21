use std::mem;

trait Trait1<T> {}
trait Trait2<'a> {
  type Ty;
}

fn _ice(param: Box<dyn for <'a> Trait1<<() as Trait2<'a>>::Ty>>) {
    //~^ ERROR the trait bound `for<'a> (): Trait2<'a>` is not satisfied
    let _e: (usize, usize) = unsafe{mem::transmute(param)};
}

trait Lifetime<'a> {
    type Out;
}
impl<'a> Lifetime<'a> for () {
    type Out = &'a ();
}
fn foo<'a>(x: &'a ()) -> <() as Lifetime<'a>>::Out {
    x
}

fn takes_lifetime(_f: for<'a> fn(&'a ()) -> <() as Lifetime<'a>>::Out) {
}

fn main() {
    takes_lifetime(foo);
}
