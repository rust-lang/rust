use super::*;

#[test]
fn test_lev_distance() {
    use std::char::{from_u32, MAX};
    // Test bytelength agnosticity
    for c in (0..MAX as u32).filter_map(from_u32).map(|i| i.to_string()) {
        assert_eq!(lev_distance(&c[..], &c[..]), 0);
    }

    let a = "\nMäry häd ä little lämb\n\nLittle lämb\n";
    let b = "\nMary häd ä little lämb\n\nLittle lämb\n";
    let c = "Mary häd ä little lämb\n\nLittle lämb\n";
    assert_eq!(lev_distance(a, b), 1);
    assert_eq!(lev_distance(b, a), 1);
    assert_eq!(lev_distance(a, c), 2);
    assert_eq!(lev_distance(c, a), 2);
    assert_eq!(lev_distance(b, c), 1);
    assert_eq!(lev_distance(c, b), 1);
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
