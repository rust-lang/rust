struct direct {
    f: &int;
}

struct indirect1 {
    g: fn@(direct);
}

struct indirect2 {
    g: fn@(direct/&);
}

struct indirect3 {
    g: fn@(direct/&self);
}

fn take_direct(p: direct) -> direct { p } //~ ERROR mismatched types
fn take_indirect1(p: indirect1) -> indirect1 { p }
fn take_indirect2(p: indirect2) -> indirect2 { p }
fn take_indirect3(p: indirect3) -> indirect3 { p } //~ ERROR mismatched types
fn main() {}
