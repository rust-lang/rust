// check-pass
// known-bug: #105782

// Should fail. Default items completely drop candidates instead of ambiguity,
// which is unsound during coherence, since coherence requires completeness.

#![feature(specialization)]
#![allow(incomplete_features)]

trait Default {
   type Id;
}

impl<T> Default for T {
   default type Id = T;
}

trait Overlap {
   type Assoc;
}

impl Overlap for u32 {
   type Assoc = usize;
}

impl Overlap for <u32 as Default>::Id {
   type Assoc = Box<usize>;
}

fn main() {}
