//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O
//! This fails without optimizations, so it should also fail with optimizations.
#![feature(type_alias_impl_trait)]

mod m {
    struct Fail<T>(T);
    impl<T> Fail<T> {
        const C: () = panic!(); //~ERROR: evaluation panicked: explicit panic
    }

    pub type NotCalledFn = impl Fn();

    #[inline(never)]
    fn not_called<T>() {
        if false {
            let _ = Fail::<T>::C;
        }
    }

    #[define_opaque(NotCalledFn)]
    fn mk_not_called() -> NotCalledFn {
        not_called::<i32>
    }
}

fn main() {
    // This does not involve a constant of `FnDef` type, it generates the value via unsafe
    // shenanigans instead. This ensures that we check all `FnDef` types that occur in a function,
    // not just those of constants. Furthermore the `FnDef` is behind an opaque type which bust be
    // normalized away to reveal the function type.
    if false {
        let x: m::NotCalledFn = unsafe { std::mem::transmute(()) };
        x();
    }
}
