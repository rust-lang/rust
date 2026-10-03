//@ edition:2018..2021

use core::convert::identity;

fn assert_static_fn<Ret>(_: impl Fn() -> Ret + 'static) {}

fn main() {
    let mut foo = 42;
    {
        let foo_ref: &u32 = &foo; //~ ERROR `foo` does not live long enough
        let closure = || *foo_ref; //~ ERROR closure may outlive the current function, but it borrows `foo_ref`, which is owned by the current function
        assert_static_fn::<u32>(closure);
    }

    {
        let foo_mut: &mut u32 = &mut foo; //~ ERROR cannot borrow `foo` as mutable because it is also borrowed as immutable
        let closure = || *foo_mut; //~ ERROR closure may outlive the current function, but it borrows `foo_mut`, which is owned by the current function
        assert_static_fn::<u32>(closure);
    }

    let really_large_copy_value = [42u128; 10_000];
    let closure = || {
        //~^ ERROR closure may outlive the current function
        let really_large_copy_ref = &really_large_copy_value;
        if identity(false) {
            drop(*really_large_copy_ref);
        }
    };
    assert_static_fn(closure);

    let closure = || {
        //~^ ERROR closure may outlive the current function
        let _ = &really_large_copy_value;
        if identity(false) {
            drop(really_large_copy_value);
        }
    };
    assert_static_fn(closure);

    let closure = || {
        //~^ ERROR closure may outlive the current function
        if identity(false) {
            drop(*&really_large_copy_value);
        }
    };
    assert_static_fn(closure);
}
