// check-pass

// This was an ICE, because the compiler ensures the
// function to be const when performing const checking,
// but functions marked with the attribute are not const
// *and* subject to const checking.

#![feature(staged_api)]
#![feature(const_trait_impl)]
#![stable(since = "1", feature = "foo")]

trait Tr {
    #[default_method_body_is_const]
    fn a() {}
}

fn main() {}
