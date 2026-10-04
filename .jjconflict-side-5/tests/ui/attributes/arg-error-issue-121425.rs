//@ compile-flags: -Zdeduplicate-diagnostics=yes

const N: usize = 8;
#[repr(align(N))]
//~^ ERROR: malformed `repr` attribute input
struct T;

#[repr(align('a'))]
//~^ ERROR: not an unsuffixed integer [E0589]
struct H;

#[repr(align("str"))]
//~^ ERROR: not an unsuffixed integer [E0589]
struct L;

#[repr(align())]
//~^ ERROR: malformed `repr` attribute input
struct X;

const P: usize = 8;
#[repr(packed(P))]
//~^ ERROR: malformed `repr` attribute input
struct A;

#[repr(packed())]
//~^ ERROR: malformed `repr` attribute input
struct B;

#[repr(packed)]
struct C;

fn main() {}
