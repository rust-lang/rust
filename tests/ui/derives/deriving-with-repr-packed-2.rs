// Check that deriving certain builtin traits on certain packed structs cause
// errors. To avoid potentially misaligned references, field copies must be
// used, which involves adding `T: Copy` bounds.

#[derive(Copy, Clone, Default, PartialEq, Eq)]
#[repr(packed)]
pub struct Foo<T>(T, T, T);

struct NonCopy;

fn main() {
    // This one is fine because `u32` impls `Copy`.
    let x: Foo<u32> = Foo(1, 2, 3);
    _ = x.clone();

    // This one is an error because `NonCopy` doesn't impl `Copy`.
    let x: Foo<NonCopy> = Foo(NonCopy, NonCopy, NonCopy);
    _ = x.clone();
    //~^ ERROR the method `clone` exists for struct `Foo<NonCopy>`, but its trait bounds were not satisfied
}
