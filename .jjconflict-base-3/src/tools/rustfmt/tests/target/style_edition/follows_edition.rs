// rustfmt-edition: 2024

fn build_sorted_static_get_entry_names(
    mut entries: Vec<(u8, &'static str)>,
) -> (
    impl Fn(
        AlphabeticalTraversal,
        Box<dyn dirents_sink::Sink<AlphabeticalTraversal>>,
    ) -> BoxFuture<'static, Result<Box<dyn dirents_sink::Sealed>, Status>>
    + Send
    + Sync
    + 'static
) {
}
