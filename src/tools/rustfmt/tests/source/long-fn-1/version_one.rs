// rustfmt-style_edition: 2015
// Tests that a function which is almost short enough, but not quite, gets
// formatted correctly.

impl Foo {
    fn some_input(&mut self, input: Input, input_path: Option<PathBuf>, ) -> (Input, Option<PathBuf>) {}

    fn some_inpu(&mut self, input: Input, input_path: Option<PathBuf>) -> (Input, Option<PathBuf>) {}
}

// #1843
#[allow(non_snake_case)]
pub extern "C" fn Java_com_exonum_binding_storage_indices_ValueSetIndexProxy_nativeContainsByHash() -> bool {
    false
}

// #3009
impl Something {
    fn my_function_name_is_way_to_long_but_used_as_a_case_study_or_an_example_its_fine(
) -> Result<  (), String  > {}
}
