fn main() {
    let b = Box::new(0);
    let x = &*b as *const i32; // soon-to-be dangling
    drop(b);
    let b = Box::new(0);
    let y = &*b as *const i32; // different allocation
    // We cannot compare these even though both are inbounds -- they *could* be
    // equal if memory was reused.
    assert!(x != y); //~ ERROR dangling pointer
}
