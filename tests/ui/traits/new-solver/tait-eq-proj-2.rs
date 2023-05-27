// compile-flags: -Ztrait-solver=next
// check-pass

#![feature(type_alias_impl_trait)]

// Similar to tests/ui/traits/new-solver/tait-eq-proj.rs
// but check the alias-sub relation in the other direction.

type Tait = impl Iterator<Item = impl Sized>;

fn mk<T>() -> T { todo!() }

fn a() {
    let x: Tait = mk();
    let mut array = mk();
    let mut z = IntoIterator::into_iter(array);
    z = x;
    array = [0i32; 32];
}

fn main() {}
