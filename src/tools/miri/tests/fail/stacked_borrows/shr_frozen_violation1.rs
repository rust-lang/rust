#![allow(cast_ref_to_mut)]

fn foo(x: &mut i32) -> i32 {
    *x = 5;
    unknown_code(&*x);
    *x // must return 5
}

fn main() {
    println!("{}", foo(&mut 0));
}

fn unknown_code(x: &i32) {
    unsafe {
        *(x as *const i32 as *mut i32) = 7; //~ ERROR: /write access .* only grants SharedReadOnly permission/
    }
}
