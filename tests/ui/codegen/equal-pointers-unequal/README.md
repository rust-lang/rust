See https://github.com/rust-lang/rust/issues/107975

Basically, if you have two pointers with the same address but from two different allocations,
the compiler gets confused whether their addresses are equal or not,
resulting in some self-contradictory behavior of the compiled code.

This folder contains some examples.
They all boil down to allocating a variable on the stack, taking its address,
getting rid of the variable, and then doing it all again.
This way we end up with two addresses stored in two `usize`s (`a` and `b`).
The addresses are (probably) equal but (definitely) come from two different allocations.
Logically, we would expect that exactly one of the following options holds true:
1. `a == b`
2. `a != b`
Sadly, the compiler does not always agree.

Due to Rust having at least three meaningfully different ways
to get a variable's address as an `usize`,
each example is provided in three versions, each in the corresponding subfolder:
1. `./as-cast/` for `&v as *const _ as usize`,
2. `./strict-provenance/` for `addr_of!(v).addr()`,
2. `./exposed-provenance/` for `addr_of!(v).expose_provenance()`.
