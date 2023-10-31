// check-pass

// This was an ICE, because the compiler ensures the
// function to be const when performing const checking,
// but functions marked with the attribute are not const
// *and* subject to const checking.

#![feature(staged_api)]
#![feature(const_trait_impl)]
#![stable(since = "1", feature = "foo")]

#[const_trait]
trait Tr {
    fn a() {}
}

fn main() {}
