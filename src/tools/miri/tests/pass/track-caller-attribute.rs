#![feature(core_intrinsics)]

use std::panic::Location;

#[track_caller]
fn tracked() -> &'static Location<'static> {
    Location::caller() // most importantly, we never get line 7
}

fn nested_intrinsic() -> &'static Location<'static> {
    Location::caller()
}

fn nested_tracked() -> &'static Location<'static> {
    tracked()
}

macro_rules! caller_location_from_macro {
    () => {
        core::panic::Location::caller()
    };
}

fn test_fn_ptr() {
    fn pass_to_ptr_call<T>(f: fn(T), x: T) {
        f(x);
    }

    #[track_caller]
    fn tracked_unit(_: ()) {
        let expected_line = line!() - 1;
        let location = std::panic::Location::caller();
        assert_eq!(location.file(), file!());
        assert_eq!(location.line(), expected_line, "call shims report location as fn definition");
    }

    pass_to_ptr_call(tracked_unit, ());
}

fn test_trait_obj() {
    trait Tracked {
        #[track_caller]
        fn handle(&self) -> &'static Location<'static> {
            std::panic::Location::caller()
        }
    }

    impl Tracked for () {}
    impl Tracked for u8 {}

    // Test that we get the correct location
    // even with a call through a trait object

    let tracked: &dyn Tracked = &5u8;
    let location = tracked.handle();
    let expected_line = line!() - 1;
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), expected_line);
    assert_eq!(location.column(), 28);

    const TRACKED: &dyn Tracked = &();
    let location = TRACKED.handle();
    let expected_line = line!() - 1;
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), expected_line);
    assert_eq!(location.column(), 28);
}

fn test_trait_obj2() {
    // track_caller on the impl but not the trait.
    pub trait Foo {
        fn foo(&self) -> &'static Location<'static>;
    }

    struct Bar;
    impl Foo for Bar {
        #[track_caller]
        fn foo(&self) -> &'static Location<'static> {
            std::panic::Location::caller()
        }
    }
    let expected_line = line!() - 4; // the `fn` signature above

    let f = &Bar as &dyn Foo;
    let loc = f.foo(); // trait doesn't track, so we don't point at this call site
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), expected_line);
}

fn main() {
    let location = Location::caller();
    let expected_line = line!() - 1;
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), expected_line);
    assert_eq!(location.column(), 20);

    let tracked = tracked();
    let expected_line = line!() - 1;
    assert_eq!(tracked.file(), file!());
    assert_eq!(tracked.line(), expected_line);
    assert_eq!(tracked.column(), 19);

    let nested = nested_intrinsic();
    assert_eq!(nested.file(), file!());
    assert_eq!(nested.line(), 11);
    assert_eq!(nested.column(), 5);

    let contained = nested_tracked();
    assert_eq!(contained.file(), file!());
    assert_eq!(contained.line(), 15);
    assert_eq!(contained.column(), 5);

    // `Location::caller()` in a macro should behave similarly to `file!` and `line!`,
    // i.e. point to where the macro was invoked, instead of the macro itself.
    let inmacro = caller_location_from_macro!();
    let expected_line = line!() - 1;
    assert_eq!(inmacro.file(), file!());
    assert_eq!(inmacro.line(), expected_line);
    assert_eq!(inmacro.column(), 19);

    let intrinsic = core::intrinsics::caller_location();
    let expected_line = line!() - 1;
    assert_eq!(intrinsic.file(), file!());
    assert_eq!(intrinsic.line(), expected_line);
    assert_eq!(intrinsic.column(), 21);

    test_fn_ptr();
    test_trait_obj();
    test_trait_obj2();
}
