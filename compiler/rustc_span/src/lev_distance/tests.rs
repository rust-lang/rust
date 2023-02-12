use super::*;

#[test]
fn test_lev_distance() {
    // Test bytelength agnosticity
    for c in (0..char::MAX as u32).filter_map(char::from_u32).map(|i| i.to_string()) {
        assert_eq!(lev_distance(&c[..], &c[..], usize::MAX), Some(0));
    }

    let a = "\nMäry häd ä little lämb\n\nLittle lämb\n";
    let b = "\nMary häd ä little lämb\n\nLittle lämb\n";
    let c = "Mary häd ä little lämb\n\nLittle lämb\n";
    assert_eq!(lev_distance(a, b, usize::MAX), Some(1));
    assert_eq!(lev_distance(b, a, usize::MAX), Some(1));
    assert_eq!(lev_distance(a, c, usize::MAX), Some(2));
    assert_eq!(lev_distance(c, a, usize::MAX), Some(2));
    assert_eq!(lev_distance(b, c, usize::MAX), Some(1));
    assert_eq!(lev_distance(c, b, usize::MAX), Some(1));
}

#[test]
fn test_lev_distance_limit() {
    assert_eq!(lev_distance("abc", "abcd", 1), Some(1));
    assert_eq!(lev_distance("abc", "abcd", 0), None);
    assert_eq!(lev_distance("abc", "xyz", 3), Some(3));
    assert_eq!(lev_distance("abc", "xyz", 2), None);
}

#[test]
fn test_method_name_similarity_score() {
    assert_eq!(lev_distance_with_substrings("empty", "is_empty", 1), Some(1));
    assert_eq!(lev_distance_with_substrings("shrunk", "rchunks", 2), None);
    assert_eq!(lev_distance_with_substrings("abc", "abcd", 1), Some(1));
    assert_eq!(lev_distance_with_substrings("a", "abcd", 1), None);
    assert_eq!(lev_distance_with_substrings("edf", "eq", 1), None);
    assert_eq!(lev_distance_with_substrings("abc", "xyz", 3), Some(3));
    assert_eq!(lev_distance_with_substrings("abcdef", "abcdef", 2), Some(0));
}

#[test]
fn test_find_best_match_for_name() {
    use crate::create_default_session_globals_then;
    create_default_session_globals_then(|| {
        let input = vec![Symbol::intern("aaab"), Symbol::intern("aaabc")];
        assert_eq!(
            find_best_match_for_name(&input, Symbol::intern("aaaa"), None),
            Some(Symbol::intern("aaab"))
        );

        assert_eq!(find_best_match_for_name(&input, Symbol::intern("1111111111"), None), None);

        let input = vec![Symbol::intern("AAAA")];
        assert_eq!(
            find_best_match_for_name(&input, Symbol::intern("aaaa"), None),
            Some(Symbol::intern("AAAA"))
        );

        let input = vec![Symbol::intern("AAAA")];
        assert_eq!(
            find_best_match_for_name(&input, Symbol::intern("aaaa"), Some(4)),
            Some(Symbol::intern("AAAA"))
        );

        let input = vec![Symbol::intern("a_longer_variable_name")];
        assert_eq!(
            find_best_match_for_name(&input, Symbol::intern("a_variable_longer_name"), None),
            Some(Symbol::intern("a_longer_variable_name"))
        );
    })
}
