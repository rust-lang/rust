#[repr(simd)] //~ ERROR: SIMD types are experimental
struct Foo([u64; 2]);

#[repr(C)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
#[repr(simd)] //~ ERROR: SIMD types are experimental
struct Bar([u64; 2]);

#[repr(simd)] //~ ERROR: SIMD types are experimental
//~^ ERROR: attribute cannot be used on
union U {f: u32}

#[repr(simd)] //~ ERROR: SIMD types are experimental
//~^ error: attribute cannot be used on
enum E { X }

fn main() {}
