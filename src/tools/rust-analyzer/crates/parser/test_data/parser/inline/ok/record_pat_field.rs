fn foo() {
    let S { 0: 1 } = ();
    let S { x: 1 } = ();
    let S { #[cfg(any())] x: 1 } = ();
}
