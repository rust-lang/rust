//@ build-pass
//@ revisions: normal randomize-layout
//@ [randomize-layout]compile-flags: -Zrandomize-layout

enum Void {}

pub struct Struct([*const (); 0], Void);

pub enum Enum {
    Variant(Struct),
}

fn main() {}
