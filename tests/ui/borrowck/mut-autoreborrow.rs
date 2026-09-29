// Test that `&mut` gets autoreborrowed on every move as expected.
//@ run-pass

fn generic(_: impl Sized) {}
fn assert_fnmut(_: &mut impl FnMut()) {}

struct Foo<T>(T);

fn main() {
    let mut_ref = &mut ();
    generic(mut_ref);
    {
        mut_ref
    };
    let _local = mut_ref;
    let _ = || mut_ref;
    let mut _tup: (&mut (),) = (&mut (),);
    _tup.0 = mut_ref;
    Foo(mut_ref);
    Foo { 0: mut_ref };
    let mut f = || {
        let _y: &mut _ = mut_ref;
    };
    f();
    f();
    assert_fnmut(&mut f);
    generic(mut_ref);

    let mut_ref_ref = &mut &mut ();
    generic(*mut_ref_ref);
    let _local = *mut_ref_ref;
    let _ = || *mut_ref_ref;
    let mut _tup: (&mut (),) = (&mut (),);
    _tup.0 = *mut_ref_ref;
    Foo(*mut_ref_ref);
    Foo { 0: *mut_ref_ref };
    let mut f = || {
        let _y: &mut _ = *mut_ref_ref;
    };
    f();
    f();
    assert_fnmut(&mut f);
    generic(*mut_ref_ref);
}
