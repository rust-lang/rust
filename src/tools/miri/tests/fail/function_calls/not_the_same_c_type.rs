mod somewhere_else {
    #[repr(C)]
    struct CType {
        field: i32,
    }

    #[unsafe(no_mangle)]
    extern "C" fn work_with_c_type(_x: CType) {}
    //~^ERROR: parameter #1 has type somewhere_else::CType passing argument of type CType
}

// Imagine we import the function from above but we end up with a copy of the type declaration.
// We only accept this if the types are *exactly* the name, including the names of all fields.
#[repr(C)]
struct CType(i32);

extern "C" {
    fn work_with_c_type(_x: CType);
}

fn main() {
    unsafe { work_with_c_type(CType(0)) };
}
