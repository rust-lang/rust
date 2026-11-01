#[discard_params_doc]
trait Trait {
    fn foo(
        &self,
        /// some docs
        bar: String,
        /// another docs
        baz: i32,
    );
}
