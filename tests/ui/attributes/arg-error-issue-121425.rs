//@ compile-flags: -Zdeduplicate-diagnostics=yes

const N: usize = 8;
#[repr(align(N))]
//~^ ERROR: incorrect `repr(align)` attribute format
struct T;

#[repr(align('a'))]
//~^ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer [E0589]
struct H;

#[repr(align("str"))]
//~^ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer [E0589]
struct L;

#[repr(align())]
//~^ ERROR: attribute format: `align` takes exactly one argument in parentheses
struct X;

const P: usize = 8;
#[repr(packed(P))]
//~^ ERROR: attribute format: `packed` expects a literal integer as argument
struct A;

#[repr(packed())]
//~^ ERROR: attribute format: `packed` takes exactly one parenthesized argument, or no parentheses at all
struct B;

#[repr(packed)]
struct C;

fn main() {}
