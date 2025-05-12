//@ revisions: normal exhaustive_patterns never_pats
//
// This tests correct handling of empty types in exhaustiveness checking.
//
// Most of the subtlety of this file happens in scrutinee places which are not required to hold
// valid data, namely dereferences and union field accesses. In these cases, empty arms can
// generally not be omitted, except with `exhaustive_patterns` which ignores this..
#![feature(never_type)]
// This feature is useful to avoid `!` falling back to `()` all the time.
#![feature(never_type_fallback)]
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![cfg_attr(never_pats, feature(never_patterns))]
//[never_pats]~^ WARN the feature `never_patterns` is incomplete
#![allow(dead_code, unreachable_code)]
#![deny(unreachable_patterns)]

#[derive(Copy, Clone)]
enum Void {}

/// A bunch of never situations that can't be normally constructed so we take them as argument.
#[derive(Copy, Clone)]
struct NeverBundle {
    never: !,
    void: Void,
    tuple_never: (!, !),
    tuple_half_never: (u32, !),
    array_3_never: [!; 3],
    result_never: Result<!, !>,
}

/// A simplified `MaybeUninit` to test union field accesses.
#[derive(Copy, Clone)]
union Uninit<T: Copy> {
    value: T,
    uninit: (),
}

impl<T: Copy> Uninit<T> {
    fn new() -> Self {
        Self { uninit: () }
    }
}

// Simple cases of omitting empty arms, all with known_valid scrutinees.
fn basic(x: NeverBundle) {
    let never: ! = x.never;
    match never {}
    match never {
        _ => {} //~ ERROR unreachable pattern
    }
    match never {
        _x => {} //~ ERROR unreachable pattern
    }

    let ref_never: &! = &x.never;
    match ref_never {}
    //~^ ERROR non-empty
    match ref_never {
        // useful, reachable
        _ => {}
    }
    match ref_never {
        // useful, reachable
        &_ => {}
    }

    let tuple_half_never: (u32, !) = x.tuple_half_never;
    match tuple_half_never {}
    match tuple_half_never {
        (_, _) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }

    let tuple_never: (!, !) = x.tuple_never;
    match tuple_never {}
    match tuple_never {
        _ => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match tuple_never {
        (_, _) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match tuple_never.0 {}
    match tuple_never.0 {
        _ => {} //~ ERROR unreachable pattern
    }

    let res_u32_never: Result<u32, !> = Ok(0);
    match res_u32_never {}
    //~^ ERROR non-exhaustive
    match res_u32_never {
        Ok(_) => {}
    }
    match res_u32_never {
        Ok(_) => {}
        Err(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match res_u32_never {
        //~^ ERROR non-exhaustive
        Ok(0) => {}
        Err(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    let Ok(_x) = res_u32_never;
    let Ok(_x) = res_u32_never.as_ref();
    //~^ ERROR refutable
    // Non-obvious difference: here there's an implicit dereference in the patterns, which makes the
    // inner place !known_valid. `exhaustive_patterns` ignores this.
    let Ok(_x) = &res_u32_never;
    //[normal,never_pats]~^ ERROR refutable

    let result_never: Result<!, !> = x.result_never;
    match result_never {}
    match result_never {
        _ => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match result_never {
        Ok(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match result_never {
        Ok(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
        _ => {}     //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match result_never {
        Ok(_) => {}  //[exhaustive_patterns]~ ERROR unreachable pattern
        Err(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
}

// Check for a few cases that `Void` and `!` are treated the same.
fn void_same_as_never(x: NeverBundle) {
    unsafe {
        match x.void {}
        match x.void {
            _ => {} //~ ERROR unreachable pattern
        }
        match x.void {
            _ if false => {} //~ ERROR unreachable pattern
        }
        let opt_void: Option<Void> = None;
        match opt_void {
            None => {}
        }
        match opt_void {
            None => {}
            Some(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
        }
        match opt_void {
            None => {}
            _ => {} //[exhaustive_patterns]~ ERROR unreachable pattern
        }

        let ref_void: &Void = &x.void;
        match *ref_void {}
        match *ref_void {
            _ => {}
        }
        let ref_opt_void: &Option<Void> = &None;
        match *ref_opt_void {
            //[normal,never_pats]~^ ERROR non-exhaustive
            None => {}
        }
        match *ref_opt_void {
            None => {}
            Some(_) => {}
        }
        match *ref_opt_void {
            None => {}
            _ => {}
        }
        match *ref_opt_void {
            None => {}
            _a => {}
        }
        let union_void = Uninit::<Void>::new();
        match union_void.value {}
        match union_void.value {
            _ => {}
        }
        let ptr_void: *const Void = std::ptr::null();
        match *ptr_void {}
        match *ptr_void {
            _ => {}
        }
    }
}

// Test if we correctly determine validity from the scrutinee expression.
fn invalid_scrutinees(x: NeverBundle) {
    let ptr_never: *const ! = std::ptr::null();
    let never: ! = x.never;
    let ref_never: &! = &never;

    struct NestedNeverBundle(NeverBundle);
    let nested_x = NestedNeverBundle(x);

    // These should be considered known_valid and warn unreachable.
    unsafe {
        // A plain `!` value must be valid.
        match never {}
        match never {
            _ => {} //~ ERROR unreachable pattern
        }
        // A block forces a copy.
        match { *ptr_never } {}
        match { *ptr_never } {
            _ => {} //~ ERROR unreachable pattern
        }
        // This field access is not a dereference.
        match x.never {}
        match x.never {
            _ => {} //~ ERROR unreachable pattern
        }
        // This nested field access is not a dereference.
        match nested_x.0.never {}
        match nested_x.0.never {
            _ => {} //~ ERROR unreachable pattern
        }
        // Indexing is like a field access. This one does not access behind a reference.
        let array_3_never: [!; 3] = x.array_3_never;
        match array_3_never[0] {}
        match array_3_never[0] {
            _ => {} //~ ERROR unreachable pattern
        }
    }

    // These should be considered !known_valid and not warn unreachable.
    unsafe {
        // A pointer may point to a place with an invalid value.
        match *ptr_never {}
        match *ptr_never {
            _ => {}
        }
        // A reference may point to a place with an invalid value.
        match *ref_never {}
        match *ref_never {
            _ => {}
        }
        // This field access is a dereference.
        let ref_x: &NeverBundle = &x;
        match ref_x.never {}
        match ref_x.never {
            _ => {}
        }
        // This nested field access is a dereference.
        let nested_ref_x: &NestedNeverBundle = &nested_x;
        match nested_ref_x.0.never {}
        match nested_ref_x.0.never {
            _ => {}
        }
        // A cast does not load.
        match (*ptr_never as Void) {}
        match (*ptr_never as Void) {
            _ => {}
        }
        // A union field may contain invalid data.
        let union_never = Uninit::<!>::new();
        match union_never.value {}
        match union_never.value {
            _ => {}
        }
        // Indexing is like a field access. This one accesses behind a reference.
        let slice_never: &[!] = &[];
        match slice_never[0] {}
        match slice_never[0] {
            _ => {}
        }
    }
}

// Test we correctly track validity as we dig into patterns. Validity changes when we go under a
// dereference or a union field access, and it otherwise preserved.
fn nested_validity_tracking(bundle: NeverBundle) {
    let never: ! = bundle.never;
    let ref_never: &! = &never;
    let tuple_never: (!, !) = bundle.tuple_never;
    let result_never: Result<!, !> = bundle.result_never;
    let result_never_err: Result<u8, !> = Ok(0);
    let ptr_result_never_err: *const Result<u8, !> = &result_never_err as *const _;
    let union_never = Uninit::<!>::new();

    // These should be considered known_valid and warn unreachable.
    match never {
        _ => {} //~ ERROR unreachable pattern
    }
    match tuple_never {
        (_, _) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match result_never {
        Ok(_) => {}  //[exhaustive_patterns]~ ERROR unreachable pattern
        Err(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }

    // These should be considered !known_valid and not warn unreachable.
    unsafe {
        match *ptr_result_never_err {
            Ok(_) => {}
            Err(_) => {}
        }
        let Ok(_) = *ptr_result_never_err; //[normal,never_pats]~ ERROR refutable pattern
    }
    match ref_never {
        &_ => {}
    }
    match union_never {
        Uninit { value: _ } => {}
    }
}

// Test we don't allow empty matches on empty types if the scrutinee is `!known_valid`.
fn invalid_empty_match(bundle: NeverBundle) {
    // We allow these two for backwards-compability.
    let x: &! = &bundle.never;
    match *x {}
    let x: &Void = &bundle.void;
    match *x {}

    let x: &(u32, !) = &bundle.tuple_half_never;
    match *x {} //[normal,never_pats]~ ERROR non-exhaustive
    let x: &(!, !) = &bundle.tuple_never;
    match *x {} //[normal,never_pats]~ ERROR non-exhaustive
    let x: &Result<!, !> = &bundle.result_never;
    match *x {} //[normal,never_pats]~ ERROR non-exhaustive
    let x: &[!; 3] = &bundle.array_3_never;
    match *x {} //[normal,never_pats]~ ERROR non-exhaustive
}

fn arrays_and_slices(x: NeverBundle) {
    let slice_never: &[!] = &[];
    match slice_never {}
    //~^ ERROR non-empty
    match slice_never {
        //[normal,never_pats]~^ ERROR not covered
        [] => {}
    }
    match slice_never {
        [] => {}
        [_] => {}
        [_, _, ..] => {}
    }
    match slice_never {
        //[normal]~^ ERROR `&[]`, `&[_]` and `&[_, _]` not covered
        //[exhaustive_patterns]~^^ ERROR `&[]` not covered
        //[never_pats]~^^^ ERROR `&[]`, `&[!]` and `&[!, !]` not covered
        [_, _, _, ..] => {}
    }
    match slice_never {
        [] => {}
        _ => {}
    }
    match slice_never {
        [] => {}
        _x => {}
    }
    match slice_never {
        //[normal]~^ ERROR `&[]` and `&[_, ..]` not covered
        //[exhaustive_patterns]~^^ ERROR `&[]` not covered
        //[never_pats]~^^^ ERROR `&[]` and `&[!, ..]` not covered
        &[..] if false => {}
    }

    match *slice_never {}
    //~^ ERROR non-empty
    match *slice_never {
        _ => {}
    }

    let array_3_never: [!; 3] = x.array_3_never;
    match array_3_never {}
    match array_3_never {
        _ => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match array_3_never {
        [_, _, _] => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match array_3_never {
        [_, ..] => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }

    let ref_array_3_never: &[!; 3] = &array_3_never;
    match ref_array_3_never {
        // useful, reachable
        &[_, _, _] => {}
    }
    match ref_array_3_never {
        // useful, !reachable
        &[_x, _, _] => {}
    }

    let array_0_never: [!; 0] = [];
    match array_0_never {}
    //~^ ERROR non-empty
    match array_0_never {
        [] => {}
    }
    match array_0_never {
        [] => {}
        _ => {} //~ ERROR unreachable pattern
    }
    match array_0_never {
        //~^ ERROR `[]` not covered
        [..] if false => {}
    }
}

// The difference between `_` and `_a` patterns is that `_a` loads the value. In case of an empty
// type, this asserts validity of the value, and thus the binding is unreachable. We don't yet
// distinguish these cases since we don't lint "unreachable" on `useful && !reachable` arms.
// Once/if never patterns are a thing, we can warn that the `_a` cases should be never patterns.
fn bindings(x: NeverBundle) {
    let opt_never: Option<!> = None;
    let ref_never: &! = &x.never;
    let ref_opt_never: &Option<!> = &None;

    // On a known_valid place.
    match opt_never {
        None => {}
        // !useful, !reachable
        Some(_) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match opt_never {
        None => {}
        // !useful, !reachable
        Some(_a) => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match opt_never {
        None => {}
        // !useful, !reachable
        _ => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }
    match opt_never {
        None => {}
        // !useful, !reachable
        _a => {} //[exhaustive_patterns]~ ERROR unreachable pattern
    }

    // The scrutinee is known_valid, but under the `&` isn't anymore.
    match ref_never {
        // useful, reachable
        _ => {}
    }
    match ref_never {
        // useful, reachable
        &_ => {}
    }
    match ref_never {
        // useful, reachable
        _a => {}
    }
    match ref_never {
        // useful, !reachable
        &_a => {}
    }
    match ref_opt_never {
        //[normal,never_pats]~^ ERROR non-exhaustive
        &None => {}
    }
    match ref_opt_never {
        &None => {}
        // useful, reachable
        _ => {}
    }
    match ref_opt_never {
        &None => {}
        // useful, reachable
        _a => {}
    }
    match ref_opt_never {
        &None => {}
        // useful, reachable
        &_ => {}
    }
    match ref_opt_never {
        &None => {}
        // useful, !reachable
        &_a => {}
    }

    // On a !known_valid place.
    match *ref_never {}
    match *ref_never {
        // useful, reachable
        _ => {}
    }
    match *ref_never {
        // useful, !reachable
        _a => {}
    }
    // This is equivalent to `match ref_never { _a => {} }`. In other words, it asserts validity of
    // `ref_never` but says nothing of the data at `*ref_never`.
    match *ref_never {
        // useful, reachable
        ref _a => {}
    }
    match *ref_opt_never {
        //[normal,never_pats]~^ ERROR non-exhaustive
        None => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, reachable
        Some(_) => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, !reachable
        Some(_a) => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, reachable
        _ => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, !reachable
        _a => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, !reachable
        _a @ Some(_) => {}
    }
    // This is equivalent to `match ref_opt_never { None => {}, _a => {} }`. In other words, it
    // asserts validity of `ref_opt_never` but says nothing of the data at `*ref_opt_never`.
    match *ref_opt_never {
        None => {}
        // useful, reachable
        ref _a => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, reachable
        ref _a @ Some(_) => {}
    }
    match *ref_opt_never {
        None => {}
        // useful, !reachable
        ref _a @ Some(_b) => {}
    }

    let ref_res_never: &Result<!, !> = &x.result_never;
    match *ref_res_never {
        //[normal,never_pats]~^ ERROR non-exhaustive
        // useful, reachable
        Ok(_) => {}
    }
    match *ref_res_never {
        // useful, reachable
        Ok(_) => {}
        // useful, reachable
        _ => {}
    }
    match *ref_res_never {
        //[normal,never_pats]~^ ERROR non-exhaustive
        // useful, !reachable
        Ok(_a) => {}
    }
    match *ref_res_never {
        // useful, !reachable
        Ok(_a) => {}
        // useful, reachable
        _ => {}
    }
    match *ref_res_never {
        // useful, !reachable
        Ok(_a) => {}
        // useful, reachable
        Err(_) => {}
    }

    let ref_tuple_half_never: &(u32, !) = &x.tuple_half_never;
    match *ref_tuple_half_never {}
    //[normal,never_pats]~^ ERROR non-empty
    match *ref_tuple_half_never {
        // useful, reachable
        (_, _) => {}
    }
    match *ref_tuple_half_never {
        // useful, reachable
        (_x, _) => {}
    }
    match *ref_tuple_half_never {
        // useful, !reachable
        (_, _x) => {}
    }
    match *ref_tuple_half_never {
        // useful, !reachable
        (0, _x) => {}
        // useful, reachable
        (1.., _) => {}
    }
}

// When we execute the condition for a guard we loads from all bindings. This asserts validity at
// all places with bindings. Surprisingly this can make subsequent arms unreachable. We choose to
// not detect this in exhaustiveness because this is rather subtle. With never patterns, we would
// recommend using a never pattern instead.
fn guards_and_validity(x: NeverBundle) {
    let never: ! = x.never;
    let ref_never: &! = &never;

    // Basic guard behavior when known_valid.
    match never {}
    match never {
        _ => {} //~ ERROR unreachable pattern
    }
    match never {
        _x => {} //~ ERROR unreachable pattern
    }
    match never {
        _ if false => {} //~ ERROR unreachable pattern
    }
    match never {
        _x if false => {} //~ ERROR unreachable pattern
    }

    // If the pattern under the guard doesn't load, all is normal.
    match *ref_never {
        // useful, reachable
        _ if false => {}
        // useful, reachable
        _ => {}
    }

    // Now the madness commences. The guard caused a load of the value thus asserting validity. So
    // there's no invalid value for `_` to catch. So the second pattern is unreachable despite the
    // guard not being taken.
    match *ref_never {
        // useful, !reachable
        _a if false => {}
        // !useful, !reachable
        _ => {}
    }
    // The above still applies to the implicit `_` pattern used for exhaustiveness.
    match *ref_never {
        // useful, !reachable
        _a if false => {}
    }
    match ref_never {
        //[normal,never_pats]~^ ERROR non-exhaustive
        // useful, !reachable
        &_a if false => {}
    }

    // Same but with subpatterns.
    let ref_result_never: &Result<!, !> = &x.result_never;
    match *ref_result_never {
        // useful, !reachable
        Ok(_x) if false => {}
        // !useful, !reachable
        Ok(_) => {}
        // useful, !reachable
        Err(_) => {}
    }
    match *ref_result_never {
        //[normal]~^ ERROR `Ok(_)` not covered
        //[never_pats]~^^ ERROR `Ok(!)` not covered
        // useful, reachable
        Ok(_) if false => {}
        // useful, reachable
        Err(_) => {}
    }
    let ref_tuple_never: &(!, !) = &x.tuple_never;
    match *ref_tuple_never {
        // useful, !reachable
        (_, _x) if false => {}
        // !useful, !reachable
        (_, _) => {}
    }
}

fn diagnostics_subtlety(x: NeverBundle) {
    // Regression test for diagnostics: don't report `Some(Ok(_))` and `Some(Err(_))`.
    let x: &Option<Result<!, !>> = &None;
    match *x {
        //[normal]~^ ERROR `Some(_)` not covered
        //[never_pats]~^^ ERROR `Some(!)` not covered
        None => {}
    }
}

fn main() {}
