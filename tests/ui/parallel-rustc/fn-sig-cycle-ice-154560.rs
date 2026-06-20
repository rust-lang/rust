// Regression test for ICE from issue #154056.

//@ ignore-parallel-frontend query cycle + ICE

#![feature(min_generic_const_args)]
#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): Send>;
    //~^ ERROR cycle detected when resolving lifetimes for `IntFactory::stream`
}
trait SendIntFactory: IntFactory<stream(..): Send> + Send {}

fn main() {}
