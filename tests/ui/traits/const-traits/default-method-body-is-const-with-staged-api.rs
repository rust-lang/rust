//@ check-pass

// This was an ICE, because the compiler ensures the
// function to be const when performing const checking,
// but functions marked with the attribute are not const
// *and* subject to const checking.

#![feature(staged_api)]
#![feature(const_trait_impl)]
#![stable(feature = "foo", since = "3.3.3")]

#[const_trait]
trait Tr {
    fn a() {}
}

fn main() {}
