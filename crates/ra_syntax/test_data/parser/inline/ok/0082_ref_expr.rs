fn foo() {
    let _ = &1;
    let _ = &mut &f();
    let _ = &1 as *const i32;
}
