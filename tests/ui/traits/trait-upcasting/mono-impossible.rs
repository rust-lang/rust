//@ build-pass

trait Supertrait<T> {
    fn method(&self) {}
}
impl<T> Supertrait<T> for () {}

trait WithAssoc {
    type Assoc;
}
trait Trait<P: WithAssoc>: Supertrait<P::Assoc> + Supertrait<()> {}

fn upcast<P>(x: &dyn Trait<P>) -> &dyn Supertrait<()> {
    x
}

fn call<P>(x: &dyn Trait<P>) {
    x.method();
}

fn main() {
    println!("{:p}", upcast::<()> as *const ());
    println!("{:p}", call::<()> as *const ());
}
