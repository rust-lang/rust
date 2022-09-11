impl Drop for u32 {} //~ ERROR E0117
//~| ERROR the `Drop` trait may only be implemented for local structs, enums, and unions

fn main() {}
