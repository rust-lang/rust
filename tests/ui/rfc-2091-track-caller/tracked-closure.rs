// run-pass

#![feature(stmt_expr_attributes)]
#![feature(closure_track_caller)]
#![feature(generator_trait)]
#![feature(generators)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;
use std::panic::Location;

type Loc = &'static Location<'static>;

#[track_caller]
fn mono_invoke_fn<F: Fn(&'static str, bool) -> (&'static str, bool, Loc)>(
    val: &F
) -> (&'static str, bool, Loc) {
    val("from_mono", false)
}

#[track_caller]
fn mono_invoke_fn_once<F: FnOnce(&'static str, bool) -> (&'static str, bool, Loc)>(
    val: F
) -> (&'static str, bool, Loc) {
    val("from_mono", false)
}

#[track_caller]
fn dyn_invoke_fn_mut(
    val: &mut dyn FnMut(&'static str, bool) -> (&'static str, bool, Loc)
) -> (&'static str, bool, Loc) {
    val("from_dyn", false)
}

#[track_caller]
fn dyn_invoke_fn_once(
    val: Box<dyn FnOnce(&'static str, bool) -> (&'static str, bool, Loc)>
) -> (&'static str, bool, Loc) {
    val("from_dyn", false)
}


fn test_closure() {
    let mut track_closure = #[track_caller] |first: &'static str, second: bool| {
        (first, second, Location::caller())
    };
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


#[track_caller]
fn mono_generator<F: Generator<String, Yield = (&'static str, String, Loc), Return = ()>>(
    val: Pin<&mut F>
) -> (&'static str, String, Loc) {
    match val.resume("Mono".to_string()) {
        GeneratorState::Yielded(val) => val,
        _ => unreachable!()
    }
}

#[track_caller]
fn dyn_generator(
    val: Pin<&mut dyn Generator<String, Yield = (&'static str, String, Loc), Return = ()>>
) -> (&'static str, String, Loc) {
    match val.resume("Dyn".to_string()) {
        GeneratorState::Yielded(val) => val,
        _ => unreachable!()
    }
}

fn test_generator() {
    let generator = #[track_caller] |arg: String| {
        yield ("first", arg.clone(), Location::caller());
        yield ("second", arg.clone(), Location::caller());
    };

    let mut pinned = Box::pin(generator);
    let (dyn_ret, dyn_arg, dyn_loc) = dyn_generator(pinned.as_mut());
    assert_eq!(dyn_ret, "first");
    assert_eq!(dyn_arg, "Dyn".to_string());
    // The `Generator` trait does not have `#[track_caller]` on `resume`, so
    // this will not match.
    assert_ne!(dyn_loc.file(), file!());


    let (mono_ret, mono_arg, mono_loc) = mono_generator(pinned.as_mut());
    let mono_line = line!() - 1;
    assert_eq!(mono_ret, "second");
    // The generator ignores the argument to the second `resume` call
    assert_eq!(mono_arg, "Dyn".to_string());
    assert_eq!(mono_loc.file(), file!());
    assert_eq!(mono_loc.line(), mono_line);
    assert_eq!(mono_loc.column(), 42);

    let non_tracked_generator = || { yield Location::caller(); };
    let non_tracked_line = line!() - 1; // This is the line of the generator, not its caller
    let non_tracked_loc = match Box::pin(non_tracked_generator).as_mut().resume(()) {
        GeneratorState::Yielded(val) => val,
        _ => unreachable!()
    };
    assert_eq!(non_tracked_loc.file(), file!());
    assert_eq!(non_tracked_loc.line(), non_tracked_line);
    assert_eq!(non_tracked_loc.column(), 44);

}

fn main() {
    test_closure();
    test_generator();
}
