struct Heap;

struct Vec<A = Heap, T>(A, T);
//~^ ERROR generic parameters with a default must be trailing

struct Foo<A, B = Vec<C>, C>(A, B, C);
//~^ ERROR generic parameters with a default must be trailing
//~| ERROR generic parameters with a default cannot use

fn main() {}
