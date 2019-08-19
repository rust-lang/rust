#[repr] //~ ERROR malformed `repr` attribute
struct _A {}

#[repr = "B"] //~ ERROR malformed `repr` attribute
struct _B {}

#[repr = "C"] //~ ERROR malformed `repr` attribute
struct _C {}

#[repr(C)]
struct _D {}

fn main() {}
