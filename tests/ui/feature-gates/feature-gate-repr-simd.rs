#[repr(simd)] //~ error: SIMD types are experimental
struct Foo([u64; 2]);

#[repr(C)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
#[repr(simd)] //~ error: SIMD types are experimental
struct Bar([u64; 2]);

fn main() {}
