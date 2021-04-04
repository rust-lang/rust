// check-pass

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

// Given how the closure desugaring is implemented (at least at the time of writing this test),
// we don't need to truncate the captured path to a reference into a packed-struct if the field
// being referenced will be moved into the closure, since it's safe to move out a field from a
// packed-struct.
//
// However to avoid surprises for the user, or issues when the closure is
// inlined we will truncate the capture to access just the struct regardless of if the field
// might get moved into the closure.
//
// It is possible for someone to try writing the code that relies on the desugaring to access a ref
// into a packed-struct without explicity using unsafe. Here we test that the compiler warns the
// user that such an access is still unsafe.
fn test_missing_unsafe_warning_on_repr_packed() {
    #[repr(packed)]
    struct Foo { x: String }

    let foo = Foo { x: String::new() };

    let c = || {
        println!("{}", foo.x);
        //~^ WARNING: reference to packed field is unaligned
        //~| WARNING: this was previously accepted by the compiler but is being phased out
        let _z = foo.x;
    };

    c();
}

fn main() {
    test_missing_unsafe_warning_on_repr_packed();
}
