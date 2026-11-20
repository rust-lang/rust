//@ compile-flags: -Znext-solver
//@ check-pass

// Previously we resolve regions in elaborated param env when normalizing
// param env.
// Since elaborated param env is unnormalized, we got non rigid type outlive
// clause in lexical region solving.
// Type aliases didn't have this problem because we set them to rigid in
// elaborated env as a hack.

fn foo<T>()
where
    [T; 1 + 1]: 'static,
{}

fn main() {}
