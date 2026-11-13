#[macro_export]
macro_rules! reference_docs {
    () => {
        "[configration].\n\n[configuration]: https://example.com/configuration"
    };
    ($docs:literal) => {
        $docs
    };
}
