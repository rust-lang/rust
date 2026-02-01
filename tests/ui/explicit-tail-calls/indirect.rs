//@ run-pass
//@ ignore-backends: gcc
#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

// Test tail calls with `PassMode::Indirect { on_stack: false, .. }` arguments.
//
// Normally an indirect argument with `on_stack: false` would be passed as a pointer to the
// caller's stack frame. For tail calls, that would be unsound, because the caller's stack
// frame is overwritten by the callee's stack frame.
//
// The solution is to write the argument into the caller's argument place (stored somewhere further
// up the stack), and forward that place.

// A struct big enough that it is not passed via registers, so that the rust calling convention uses
// `Indirect { on_stack: false, .. }`.
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Big([u64; 4]);

#[inline(never)]
fn update_in_caller(y: Big) -> u64 {
    #[inline(never)]
    fn helper(x: Big) -> u64 {
        x.0.iter().sum()
    }

    let x = Big([y.0[0], 2, 3, 4]);

    // `x` is actually stored in `y`'s space.
    become helper(x)
}

#[inline(never)]
fn swapper<T>(a: T, b: T) -> (T, T) {
    #[inline(never)]
    fn helper<T>(a: T, b: T) -> (T, T) {
        (a, b)
    }

    become helper(b, a)
}

#[inline(never)]
fn swapper_derived(a: Big, _: (u64, u64), b: Big, _: (u64, u64)) -> ((u64, u64), (u64, u64)) {
    #[inline(never)]
    fn helper(_: Big, x: (u64, u64), _: Big, y: (u64, u64)) -> ((u64, u64), (u64, u64)) {
        (x, y)
    }

    // Read the values at various points in the swapping process, testing that they have the correct
    // value at every point.
    become helper(b, (a.0[0], b.0[0]), a, (a.0[0], b.0[0]));
}

fn main() {
    assert_eq!(update_in_caller(Big::default()), 0 + 2 + 3 + 4);

    assert_eq!(swapper(u8::MIN, u8::MAX), (u8::MAX, u8::MIN));
    // i128 uses `PassMode::Indirect { on_stack: false, .. }` on x86_64 MSVC.
    assert_eq!(swapper(i128::MIN, i128::MAX), (i128::MAX, i128::MIN));
    assert_eq!(swapper(Big([1; 4]), Big([2; 4])), (Big([2; 4]), Big([1; 4])));

    assert_eq!(swapper_derived(Big([1; 4]), (0, 0), Big([2; 4]), (0, 0)), ((1, 2), (1, 2)));
}
