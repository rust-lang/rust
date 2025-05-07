#[repr(simd)] //~ ERROR: SIMD types are experimental
struct Foo([u64; 2]);

#[repr(C)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
#[repr(simd)] //~ ERROR: SIMD types are experimental
struct Bar([u64; 2]);

#[repr(simd)] //~ ERROR: SIMD types are experimental
//~^ ERROR: attribute should be applied to a struct
union U {f: u32}

#[repr(simd)] //~ ERROR: SIMD types are experimental
//~^ error: attribute should be applied to a struct
enum E { X }

fn main() {}
