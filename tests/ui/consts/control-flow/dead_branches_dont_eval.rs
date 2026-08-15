//@ build-pass

// issue 122301 - currently the only way to supress
// const eval and codegen of code conditional on some other const

struct Foo<T, const N: usize>(T);

impl<T, const N: usize> Foo<T, N> {
    const BAR: () = if N == 0 {
        panic!()
    };
}

struct Invoke<T, const N: usize>(T);

impl<T, const N: usize> Invoke<T, N> {
    const FUN: fn() = if N != 0 {
        || Foo::<T, N>::BAR
    } else {
        || {}
    };
}

// without closures

struct S<T>(T);
impl<T> S<T> {
    const C: () = panic!();
}

const fn bar<T>() { S::<T>::C }

struct ConstIf<T, const N: usize>(T);

impl<T, const N: usize> ConstIf<T, N> {
    const VAL: () = if N != 0 {
        bar::<T>() // not called for N == 0, and hence not monomorphized
    } else {
        ()
    };
}

fn main() {
    let _val = Invoke::<(), 0>::FUN();
    let _val = ConstIf::<(), 0>::VAL;
}
