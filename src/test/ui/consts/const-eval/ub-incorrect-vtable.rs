// This test contains code with incorrect vtables in a const context:
// - from issue 86132: a trait object with invalid alignment caused an ICE in const eval, and now
//   triggers an error
// - a similar test that triggers a previously-untested const UB error: emitted close to the above
//   error, it checks the correctness of the size
//
// As is, this code will only hard error when the constants are used, and the errors are emitted via
// the `#[allow]`-able `const_err` lint. However, if the transparent wrapper technique to prevent
// reborrows is used -- from `ub-wide-ptr.rs` -- these two errors reach validation and would trigger
// ICEs as tracked by #86193. So we also use the transparent wrapper to verify proper validation
// errors are emitted instead of ICEs.

// stderr-per-bitwidth
// normalize-stderr-test "alloc\d+" -> "allocN"

trait Trait {}

const INVALID_VTABLE_ALIGNMENT: &dyn Trait =
    unsafe { std::mem::transmute((&92u8, &[0usize, 1usize, 1000usize])) };
//~^ ERROR evaluation of constant value failed
//~| invalid vtable: alignment `1000` is not a power of 2

const INVALID_VTABLE_SIZE: &dyn Trait =
    unsafe { std::mem::transmute((&92u8, &[1usize, usize::MAX, 1usize])) };
//~^ ERROR evaluation of constant value failed
//~| invalid vtable: size is bigger than largest supported object

#[repr(transparent)]
struct W<T>(T);

// The drop fn is checked before size/align are, so get ourselves a "sufficiently valid" drop fn
fn drop_me(_: *mut usize) {}

const INVALID_VTABLE_ALIGNMENT_UB: W<&dyn Trait> =
    unsafe { std::mem::transmute((&92u8, &(drop_me as fn(*mut usize), 1usize, 1000usize))) };
//~^^ ERROR it is undefined behavior to use this value
//~| invalid vtable: alignment `1000` is not a power of 2

const INVALID_VTABLE_SIZE_UB: W<&dyn Trait> =
    unsafe { std::mem::transmute((&92u8, &(drop_me as fn(*mut usize), usize::MAX, 1usize))) };
//~^^ ERROR it is undefined behavior to use this value
//~| invalid vtable: size is bigger than largest supported object

fn main() {}
