// This test contains code with incorrect vtables in a const context:
// - from issue 86132: a trait object with invalid alignment caused an ICE in const eval, and now
//   triggers an error
// - a similar test that triggers a previously-untested const UB error: emitted close to the above
//   error, it checks the correctness of the size

trait Trait {}

const INVALID_VTABLE_ALIGNMENT: &dyn Trait =
    unsafe { std::mem::transmute((&92u8, &[0usize, 1usize, 1000usize])) };
//~^ ERROR any use of this value will cause an error
//~| WARNING this was previously accepted by the compiler
//~| invalid vtable: alignment `1000` is not a power of 2

const INVALID_VTABLE_SIZE: &dyn Trait =
    unsafe { std::mem::transmute((&92u8, &[1usize, usize::MAX, 1usize])) };
//~^ ERROR any use of this value will cause an error
//~| WARNING this was previously accepted by the compiler
//~| invalid vtable: size is bigger than largest supported object

fn main() {}
