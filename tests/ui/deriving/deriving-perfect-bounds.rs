// Test for #26925: derive should not add unnecessary trait bounds on type
// parameters that only appear inside function pointer types. Function
// pointers always implement all derivable traits (Clone, Copy, Debug,
// PartialEq, Eq, Hash, PartialOrd, Ord) regardless of their type params.

//@ check-pass

use std::marker::PhantomData;

// Case 1: fn pointer field — fn(T) is always Clone/Copy, T doesn't need those
#[derive(Clone, Copy)]
struct FnPointer<T>(fn(T));

// Case 2: Two fn pointers — both always Clone/Copy
#[derive(Clone, Copy)]
struct TwoFnPtrs<T>(fn(T), fn(T) -> bool);

// Case 3: fn pointer returning T — fn() -> T is always Clone
#[derive(Clone, Copy)]
struct FnReturning<T>(fn() -> T);

// Case 4: PhantomData wrapping fn pointer — T only inside fn ptr
#[derive(Clone)]
struct PhantomFn<T>(PhantomData<fn() -> T>);

// Case 5: Multiple fn pointer fields
#[derive(Clone, Copy)]
struct MultiFn<T, U>(fn(T), fn(U) -> bool);

// Case 6: Enum with fn pointer variants
#[derive(Clone, Copy)]
enum FnEnum<T> {
    Handler(fn(T)),
    Mapper(fn(T) -> bool),
}

// Case 7: Debug derive with fn pointer
#[derive(Debug)]
struct DebugFn<T> {
    f: fn(T) -> bool,
}

// Case 8: PartialEq/Eq with fn pointer
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(PartialEq, Eq)]
struct CmpFn<T>(fn(T));

// Case 9: Hash with fn pointer
#[derive(Hash)]
struct HashFn<T>(fn(T) -> u64);

// Case 10: Mixed — fn(T) and concrete type (no T outside fn ptr)
#[derive(Clone)]
struct MixedFnConcrete<T> {
    callback: fn(T),
    name: String,
}

// Case 11: Nested — Vec<fn(T)> — T only inside fn ptr
// Vec<fn(T)>: Clone because fn(T): Clone always
#[derive(Clone)]
struct VecOfFn<T>(Vec<fn(T)>);

// Case 12: Direct T — T: Clone still needed (correct behavior preserved)
#[derive(Clone)]
struct Direct<T>(T);

fn assert_clone<T: Clone>() {}
fn assert_copy<T: Copy>() {}

struct NotClone;

fn main() {
    // fn pointer cases — should all work WITHOUT T: Clone
    assert_clone::<FnPointer<NotClone>>();
    assert_copy::<FnPointer<NotClone>>();
    assert_clone::<TwoFnPtrs<NotClone>>();
    assert_copy::<TwoFnPtrs<NotClone>>();
    assert_clone::<FnReturning<NotClone>>();
    assert_copy::<FnReturning<NotClone>>();
    assert_clone::<PhantomFn<NotClone>>();
    assert_clone::<MultiFn<NotClone, NotClone>>();
    assert_copy::<MultiFn<NotClone, NotClone>>();
    assert_clone::<FnEnum<NotClone>>();
    assert_copy::<FnEnum<NotClone>>();
    assert_clone::<MixedFnConcrete<NotClone>>();
    assert_clone::<VecOfFn<NotClone>>();

    // Direct<T> still requires T: Clone (correct)
    assert_clone::<Direct<String>>();
}
