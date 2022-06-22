// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

#![deny(dead_code)]

union Foo {
    x: usize,
    b: bool, //~ ERROR: field `b` is never read
    _unused: u16,
}

fn field_read(f: Foo) -> usize {
    unsafe { f.x }
}

fn main() {
    let _ = field_read(Foo { x: 2 });
}
