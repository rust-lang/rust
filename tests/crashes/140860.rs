//@ known-bug: #140860
#![feature(min_generic_const_args)]
#![feature(unsized_const_params)]
#![feature(with_negative_coherence, negative_impls)]
trait a < const b : &'static str> {} trait c {} struct d< e >(e);
impl<e> c for e where e: a<""> {}
impl<e> c for d<e> {}
impl<e> !a<f> for e {}
const f : &str = "";
fn main() {}
