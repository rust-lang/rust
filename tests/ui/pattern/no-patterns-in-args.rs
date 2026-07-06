// FIXME(fmease): Rename file name (e.g., `arg` -> `param`).
// FIXME(fmease): Add historical context.

extern "C" {
    fn f1(mut arg: u8); //~ ERROR patterns aren't allowed in foreign function declarations
    fn f2(&arg: u8); //~ ERROR patterns aren't allowed in foreign function declarations
    fn f3(arg @ _: u8); //~ ERROR patterns aren't allowed in foreign function declarations
    fn g1(arg: u8); // OK
    fn g2(_: u8); // OK
    fn g3(u8); //~ ERROR expected one of
}

type A1 = fn(mut arg: u8); //~ ERROR patterns aren't allowed in function pointer types
type A2 = fn(&arg: u8);
//~^ ERROR parameters can't have complex patterns in function pointer types
type B1 = fn(arg: u8); // OK
type B2 = fn(_: u8); // OK
type B3 = fn(u8); // OK

fn main() {}
