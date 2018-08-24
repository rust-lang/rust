impl Drop for u32 {} //~ ERROR E0117
//~| ERROR the Drop trait may only be implemented on structures
//~| implementing Drop requires a struct

fn main() {
}
