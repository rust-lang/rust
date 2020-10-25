struct Heap;

struct Vec<A = Heap, T>(A, T);
//~^ ERROR type parameters with a default must be trailing

struct Foo<A, B = Vec<C>, C>(A, B, C);
//~^ ERROR type parameters with a default must be trailing
//~| ERROR type parameters with a default cannot use forward declared identifiers

fn main() {}
