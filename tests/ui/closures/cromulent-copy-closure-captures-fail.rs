//@ revisions: edition2018 edition2021
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021..

use core::convert::identity;

fn assert_static_fn<Ret>(_: impl Fn() -> Ret + 'static) {}

fn main() {
    let closure = {
        let mut foo = 42;
        let foo_ref: &u32 = &foo; //~ ERROR `foo` does not live long enough
        || *foo_ref
    };
    closure();

    let closure = {
        let mut foo = 42;
        let foo_mut: &mut u32 = &mut foo; //~ ERROR `foo` does not live long enough
        || *foo_mut //[edition2018]~ ERROR closure may outlive the current block, but it borrows `foo_mut`, which is owned by the current block
    };
    closure();

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

    let val = 21;
    // `==` takes an implicit reference
    let closure = || val == 21;
    //~^ ERROR closure may outlive the current function
    assert_static_fn(closure);

    #[allow(unused)]
    struct Int(i32);
    #[allow(unused)]
    #[derive(Clone, Copy)]
    struct B<'a>(&'a i32);

    struct MyStruct<'a, 'b> {
        _a: &'a Int,
        b: B<'b>,
    }

    fn baz<'a, 'b, 'c>(m: &'a MyStruct<'b, 'c>) -> impl FnMut() + use<'c> {
        let c = || {
            let _unused = m.b;
        };
        c
        //~^ ERROR hidden type for `impl FnMut()` captures lifetime that does not appear in bounds
    }

    baz(&MyStruct { _a: &Int(42), b: B(&42) })();
}
