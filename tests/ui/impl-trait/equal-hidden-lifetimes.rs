// Test that we consider equal regions when checking for hidden regions in
// opaque types

// check-pass

// `'a == 'static` so `&'a i32` is fine as the return type
fn equal_regions_static<'a: 'static>(x: &'a i32) -> impl Sized {
    x
}

// `'a == 'b` so `&'b i32` is fine as the return type
fn equal_regions<'a: 'b, 'b: 'a>(x: &'b i32) -> impl Sized + 'a {
    let y: &'a i32 = x;
    let z: &'b i32 = y;
    x
}

// `'a == 'b` so `&'a i32` is fine as the return type
fn equal_regions_rev<'a: 'b, 'b: 'a>(x: &'a i32) -> impl Sized + 'b {
    let y: &'a i32 = x;
    let z: &'b i32 = y;
    x
}

// `'a == 'b` so `*mut &'b i32` is fine as the return type
fn equal_regions_inv<'a: 'b, 'b: 'a>(x: *mut &'b i32) -> impl Sized + 'a {
    let y: *mut &'a i32 = x;
    let z: *mut &'b i32 = y;
    x
}

// `'a == 'b` so `*mut &'a i32` is fine as the return type
fn equal_regions_inv_rev<'a: 'b, 'b: 'a>(x: *mut &'a i32) -> impl Sized + 'b {
    let y: *mut &'a i32 = x;
    let z: *mut &'b i32 = y;
    x
}

// Should be able to infer `fn(&'static ())` as the return type.
fn contravariant_lub<'a, 'b: 'a, 'c: 'a, 'd: 'b + 'c>(
    x: fn(&'b ()),
    y: fn(&'c ()),
    c: bool,
) -> impl Sized + 'a {
    if c { x } else { y }
}

fn main() {}
