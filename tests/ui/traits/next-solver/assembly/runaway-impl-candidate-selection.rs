//@ compile-flags: -Znext-solver

// In the new solver, we are trying to select `<?0 as Iterator>::Item: Debug`,
// which, naively can be unified with every impl of `Debug` if we're not careful.
// This test makes sure that we treat projections with inference var substs as
// placeholders during fast reject.

fn iter<T: Iterator>() -> <T as Iterator>::Item {
    todo!()
}

fn main() {
    println!("{:?}", iter::<_>());
    //~^ ERROR type annotations needed
}
