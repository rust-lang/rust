//@ known-bug: #108248
//@ needs-rustc-debug-assertions
//@ compile-flags: -Wunused-lifetimes
fn main() {
    let _: extern fn<'a: 'static>();
}
