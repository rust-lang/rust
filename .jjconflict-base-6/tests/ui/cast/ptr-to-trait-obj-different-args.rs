//@ check-fail
//
// issue: <https://github.com/rust-lang/rust/issues/120222>

trait A {}
impl<T> A for T {}
trait B {}
impl<T> B for T {}

trait Trait<G> {}
struct X;
impl<T> Trait<X> for T {}
struct Y;
impl<T> Trait<Y> for T {}

fn main() {
    let a: *const dyn A = &();
    let b: *const dyn B = a as _; //~ error: casting `*const dyn A` as `*const dyn B` is invalid

    let x: *const dyn Trait<X> = &();
    let y: *const dyn Trait<Y> = x as _; //~ error: casting `*const dyn Trait<X>` as `*const dyn Trait<Y>` is invalid

    _ = (b, y);
}

fn generic<T>(x: *const dyn Trait<X>, t: *const dyn Trait<T>) {
    let _: *const dyn Trait<T> = x as _; //~ error: casting `*const (dyn Trait<X> + 'static)` as `*const dyn Trait<T>` is invalid
    let _: *const dyn Trait<X> = t as _; //~ error: casting `*const (dyn Trait<T> + 'static)` as `*const dyn Trait<X>` is invalid
}

trait Assocked {
    type Assoc: ?Sized;
}

fn change_assoc(x: *mut dyn Assocked<Assoc = u8>) -> *mut dyn Assocked<Assoc = u32> {
    x as _ //~ error: casting `*mut (dyn Assocked<Assoc = u8> + 'static)` as `*mut (dyn Assocked<Assoc = u32> + 'static)` is invalid
}
