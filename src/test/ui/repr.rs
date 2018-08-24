// compile-pass

#[repr]
//^ WARN `repr` attribute must have a hint
struct _A {}

#[repr = "B"]
//^ WARN `repr` attribute isn't configurable with a literal
struct _B {}

#[repr = "C"]
//^ WARN `repr` attribute isn't configurable with a literal
struct _C {}

#[repr(C)]
struct _D {}

fn main() {}
