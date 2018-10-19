fn main() {
    let target = 42;
    // Make sure raw ptr with raw tag cannot mutate frozen location without breaking the shared ref.
    let r#ref = &target; // freeze
    let ptr = r#ref as *const _ as *mut _; // raw ptr, with raw tag
    unsafe { *ptr = 42; }
    let _val = *r#ref; //~ ERROR Shr reference with non-reactivatable tag Frz
}
