// Regression test for ICE from issue #154056.

//@ check-pass

#![feature(min_generic_const_args)]
#![feature(return_type_notation)]

trait IntFactory {
    fn stream(&self) -> impl IntFactory<stream(..): Send>;
}

trait SendIntFactory: IntFactory<stream(..): Send> + Send {}

fn main() {}
