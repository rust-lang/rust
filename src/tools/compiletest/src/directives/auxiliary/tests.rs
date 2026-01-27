use super::*;

#[test]
fn test_aux_crate_value_no_modifiers() {
    assert_eq!(
        AuxCrate { extern_modifiers: None, name: "foo".to_string(), path: "foo.rs".to_string() },
        parse_aux_crate("foo=foo.rs".to_string())
    );
}

#[test]
fn test_aux_crate_value_with_modifiers() {
    assert_eq!(
        AuxCrate {
            extern_modifiers: Some("noprelude".to_string()),
            name: "foo".to_string(),
            path: "foo.rs".to_string()
        },
        parse_aux_crate("noprelude:foo=foo.rs".to_string())
    );
}

#[test]
#[should_panic(expected = "couldn't parse aux-crate value `foo.rs` (should be e.g. `log=log.rs`)")]
fn test_aux_crate_value_invalid() {
    parse_aux_crate("foo.rs".to_string());
}
