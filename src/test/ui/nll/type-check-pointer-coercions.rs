#![feature(nll)]

fn shared_to_const<'a, 'b>(x: &&'a i32) -> *const &'b i32 {
    x   //~ ERROR
}

fn unique_to_const<'a, 'b>(x: &mut &'a i32) -> *const &'b i32 {
    x   //~ ERROR
}

fn unique_to_mut<'a, 'b>(x: &mut &'a i32) -> *mut &'b i32 {
    // Two errors because *mut is invariant
    x   //~ ERROR
        //~| ERROR
}

fn mut_to_const<'a, 'b>(x: *mut &'a i32) -> *const &'b i32 {
    x   //~ ERROR
}

fn array_elem<'a, 'b>(x: &'a i32) -> *const &'b i32 {
    let z = &[x; 3];
    let y = z as *const &i32;
    y   //~ ERROR
}

fn array_coerce<'a, 'b>(x: &'a i32) -> *const [&'b i32; 3] {
    let z = &[x; 3];
    let y = z as *const [&i32; 3];
    y   //~ ERROR
}

fn nested_array<'a, 'b>(x: &'a i32) -> *const [&'b i32; 2] {
    let z = &[[x; 2]; 3];
    let y = z as *const [&i32; 2];
    y   //~ ERROR
}

fn main() {}
