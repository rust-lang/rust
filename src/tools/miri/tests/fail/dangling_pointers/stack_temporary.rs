// This should fail even without validation
//@compile-flags: -Zmiri-disable-validation

unsafe fn make_ref<'a>(x: *mut i32) -> &'a mut i32 {
    &mut *x
}

fn main() {
    unsafe {
        let x = make_ref(&mut 0); // The temporary storing "0" is deallocated at the ";"!
        let val = *x; //~ ERROR: dereferenced after this allocation got freed
        println!("{}", val);
    }
}
