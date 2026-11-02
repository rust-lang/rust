//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
// issue: rust-lang/rust#101852
// ICE opaque type with non-universal region substs

pub fn ice(x: impl AsRef<str>) -> impl IntoIterator<Item = ()> {
//~^ WARN function cannot return without recursing
    vec![].append(&mut ice(x.as_ref()));
    //[current]~^ ERROR expected generic type parameter, found `&str`

    Vec::new()
}

fn main() {}
