#[repr]
//~^ ERROR attribute must be of the form
struct _A {}

#[repr = "B"]
//~^ ERROR attribute must be of the form
struct _B {}

#[repr = "C"]
//~^ ERROR attribute must be of the form
struct _C {}

#[repr(C)]
struct _D {}

fn main() {}
