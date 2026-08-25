fn foo() {
    // reference operator
    let _ = &1;
    let _ = &mut &f();
    let _ = &raw;
    let _ = &raw.0;
    // raw reference operator
    let _ = &raw mut foo;
    let _ = &raw const foo;
    let _ = &raw foo;
}
