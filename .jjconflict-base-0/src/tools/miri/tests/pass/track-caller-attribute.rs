#![feature(core_intrinsics)]
#![feature(stmt_expr_attributes)]
#![feature(closure_track_caller)]
#![feature(coroutine_trait)]
#![feature(coroutines)]

use std::ops::{Coroutine, CoroutineState};
use std::panic::Location;
use std::pin::Pin;

type Loc = &'static Location<'static>;

#[track_caller]
fn tracked() -> Loc {
    Location::caller() // most importantly, we never get line 7
}

fn nested_intrinsic() -> Loc {
    Location::caller()
}

fn nested_tracked() -> Loc {
    tracked()
}

macro_rules! caller_location_from_macro {
    () => {
        core::panic::Location::caller()
    };
}

fn test_basic() {
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
    assert_eq!(nested.line(), 19);
    assert_eq!(nested.column(), 5);

    let contained = nested_tracked();
    assert_eq!(contained.file(), file!());
    assert_eq!(contained.line(), 23);
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

fn test_closure() {
    #[track_caller]
    fn mono_invoke_fn<F: Fn(&'static str, bool) -> (&'static str, bool, Loc)>(
        val: &F,
    ) -> (&'static str, bool, Loc) {
        val("from_mono", false)
    }

    #[track_caller]
    fn mono_invoke_fn_once<F: FnOnce(&'static str, bool) -> (&'static str, bool, Loc)>(
        val: F,
    ) -> (&'static str, bool, Loc) {
        val("from_mono", false)
    }

    #[track_caller]
    fn dyn_invoke_fn_mut(
        val: &mut dyn FnMut(&'static str, bool) -> (&'static str, bool, Loc),
    ) -> (&'static str, bool, Loc) {
        val("from_dyn", false)
    }

    #[track_caller]
    fn dyn_invoke_fn_once(
        val: Box<dyn FnOnce(&'static str, bool) -> (&'static str, bool, Loc)>,
    ) -> (&'static str, bool, Loc) {
        val("from_dyn", false)
    }

    let mut track_closure = #[track_caller]
    |first: &'static str, second: bool| (first, second, Location::caller());
    let (first_arg, first_bool, first_loc) = track_closure("first_arg", true);
    let first_line = line!() - 1;
    assert_eq!(first_arg, "first_arg");
    assert_eq!(first_bool, true);
    assert_eq!(first_loc.file(), file!());
    assert_eq!(first_loc.line(), first_line);
    assert_eq!(first_loc.column(), 46);

    let (dyn_arg, dyn_bool, dyn_loc) = dyn_invoke_fn_mut(&mut track_closure);
    assert_eq!(dyn_arg, "from_dyn");
    assert_eq!(dyn_bool, false);
    // `FnMut::call_mut` does not have `#[track_caller]`,
    // so this will not match
    assert_ne!(dyn_loc.file(), file!());

    let (dyn_arg, dyn_bool, dyn_loc) = dyn_invoke_fn_once(Box::new(track_closure));
    assert_eq!(dyn_arg, "from_dyn");
    assert_eq!(dyn_bool, false);
    // `FnOnce::call_once` does not have `#[track_caller]`
    // so this will not match
    assert_ne!(dyn_loc.file(), file!());

    let (mono_arg, mono_bool, mono_loc) = mono_invoke_fn(&track_closure);
    let mono_line = line!() - 1;
    assert_eq!(mono_arg, "from_mono");
    assert_eq!(mono_bool, false);
    assert_eq!(mono_loc.file(), file!());
    assert_eq!(mono_loc.line(), mono_line);
    assert_eq!(mono_loc.column(), 43);

    let (mono_arg, mono_bool, mono_loc) = mono_invoke_fn_once(track_closure);
    let mono_line = line!() - 1;
    assert_eq!(mono_arg, "from_mono");
    assert_eq!(mono_bool, false);
    assert_eq!(mono_loc.file(), file!());
    assert_eq!(mono_loc.line(), mono_line);
    assert_eq!(mono_loc.column(), 43);

    let non_tracked_caller = || Location::caller();
    let non_tracked_line = line!() - 1; // This is the line of the closure, not its caller
    let non_tracked_loc = non_tracked_caller();
    assert_eq!(non_tracked_loc.file(), file!());
    assert_eq!(non_tracked_loc.line(), non_tracked_line);
    assert_eq!(non_tracked_loc.column(), 33);
}

fn test_coroutine() {
    #[track_caller]
    fn mono_coroutine<F: Coroutine<String, Yield = (&'static str, String, Loc), Return = ()>>(
        val: Pin<&mut F>,
    ) -> (&'static str, String, Loc) {
        match val.resume("Mono".to_string()) {
            CoroutineState::Yielded(val) => val,
            _ => unreachable!(),
        }
    }

    #[track_caller]
    fn dyn_coroutine(
        val: Pin<&mut dyn Coroutine<String, Yield = (&'static str, String, Loc), Return = ()>>,
    ) -> (&'static str, String, Loc) {
        match val.resume("Dyn".to_string()) {
            CoroutineState::Yielded(val) => val,
            _ => unreachable!(),
        }
    }

    #[rustfmt::skip]
    let coroutine = #[track_caller] #[coroutine] |arg: String| {
        yield ("first", arg.clone(), Location::caller());
        yield ("second", arg.clone(), Location::caller());
    };

    let mut pinned = Box::pin(coroutine);
    let (dyn_ret, dyn_arg, dyn_loc) = dyn_coroutine(pinned.as_mut());
    assert_eq!(dyn_ret, "first");
    assert_eq!(dyn_arg, "Dyn".to_string());
    // The `Coroutine` trait does not have `#[track_caller]` on `resume`, so
    // this will not match.
    assert_ne!(dyn_loc.file(), file!());

    let (mono_ret, mono_arg, mono_loc) = mono_coroutine(pinned.as_mut());
    let mono_line = line!() - 1;
    assert_eq!(mono_ret, "second");
    // The coroutine ignores the argument to the second `resume` call
    assert_eq!(mono_arg, "Dyn".to_string());
    assert_eq!(mono_loc.file(), file!());
    assert_eq!(mono_loc.line(), mono_line);
    assert_eq!(mono_loc.column(), 42);

    #[rustfmt::skip]
    let non_tracked_coroutine = #[coroutine] || { yield Location::caller(); };
    let non_tracked_line = line!() - 1; // This is the line of the coroutine, not its caller
    let non_tracked_loc = match Box::pin(non_tracked_coroutine).as_mut().resume(()) {
        CoroutineState::Yielded(val) => val,
        _ => unreachable!(),
    };
    assert_eq!(non_tracked_loc.file(), file!());
    assert_eq!(non_tracked_loc.line(), non_tracked_line);
    assert_eq!(non_tracked_loc.column(), 57);
}

fn main() {
    test_basic();
    test_fn_ptr();
    test_trait_obj();
    test_trait_obj2();
    test_closure();
    test_coroutine();
}
