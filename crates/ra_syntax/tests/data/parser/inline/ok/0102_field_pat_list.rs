fn foo() {
    let S {} = ();
    let S { f, ref mut g } = ();
    let S { h: _, ..} = ();
    let S { h: _, } = ();
}
