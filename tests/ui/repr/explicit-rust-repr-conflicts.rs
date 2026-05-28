#[repr(C, Rust)] //~ ERROR conflicting representation hints
struct S {
    a: i32,
}


#[repr(Rust)] //~ ERROR conflicting representation hints
#[repr(C)]
struct T {
    a: i32,
}

#[repr(Rust, u64)] //~ ERROR conflicting representation hints
enum U {
    V,
}

#[repr(Rust, simd)]
//~^ ERROR conflicting representation hints
//~| ERROR SIMD types are experimental and possibly buggy
struct F32x4([f32; 4]);

fn main() {}
