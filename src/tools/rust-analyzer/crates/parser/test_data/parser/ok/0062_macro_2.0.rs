macro parse_use_trees($($s:expr),* $(,)*) {
    vec![
        $(parse_use_tree($s),)*
    ]
}

#[test]
fn test_use_tree_merge() {
    macro test_merge([$($input:expr),* $(,)*], [$($output:expr),* $(,)*]) {
        assert_eq!(
            merge_use_trees(parse_use_trees!($($input,)*)),
            parse_use_trees!($($output,)*),
        );
    }
}
