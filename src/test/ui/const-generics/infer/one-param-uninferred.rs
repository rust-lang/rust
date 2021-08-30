// Test that we emit an error if we cannot properly infer a constant.
fn foo<const N: usize, const M: usize>() -> [u8; N] {
    todo!()
}

fn main() {
    // FIXME(const_generics): Currently this only suggests one const parameter,
    // but instead it should suggest to provide all parameters.
    let _: [u8; 17] = foo();
                  //~^ ERROR type annotations needed
}
