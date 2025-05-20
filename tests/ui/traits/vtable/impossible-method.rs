//@ run-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Id {
    type This<'a>;
}
impl<T> Id for T {
    type This<'a> = T;
}

trait Trait<T> {}
impl<T: Id> Trait<for<'a> fn(T::This<'a>)> for T {}

trait Method<T: Id> {
    fn call_me(&self)
    where
        T: Trait<for<'a> fn(T::This<'a>)>;
}

impl<T, U> Method<U> for T {
    fn call_me(&self) {
        println!("method was reachable");
    }
}

fn generic<T: Id>(x: &dyn Method<T>) {
    // Proving `T: Trait<for<'a> fn(T::This<'a>)>` holds.
    x.call_me();
}

fn main() {
    // Proving `u32: Trait<fn(u32)>` fails due to incompleteness.
    // We don't add the method to the vtable of `dyn Method`, so
    // calling it causes UB.
    generic::<u32>(&());
}
