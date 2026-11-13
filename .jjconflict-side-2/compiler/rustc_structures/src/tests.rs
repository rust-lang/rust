use super::*;

#[test]
fn test_combine_with_defaults_no_conflict() {
    let defaults = SanitizerSet::SHADOWCALLSTACK;
    let explicit = SanitizerSet::ADDRESS;
    assert_eq!(
        explicit.combine_with_defaults(defaults),
        SanitizerSet::ADDRESS | SanitizerSet::SHADOWCALLSTACK
    );
}

#[test]
fn test_combine_with_defaults_safestack_address_conflict() {
    let defaults = SanitizerSet::SAFESTACK;
    let explicit = SanitizerSet::ADDRESS;
    // SafeStack should be implicitly disabled when Address is explicitly provided.
    assert_eq!(explicit.combine_with_defaults(defaults), SanitizerSet::ADDRESS);
}

#[test]
fn test_combine_with_defaults_empty_explicit() {
    let defaults = SanitizerSet::SAFESTACK;
    let explicit = SanitizerSet::empty();
    assert_eq!(explicit.combine_with_defaults(defaults), SanitizerSet::SAFESTACK);
}

#[test]
fn test_combine_with_defaults_safestack_cfi() {
    let defaults = SanitizerSet::SAFESTACK;
    let explicit = SanitizerSet::CFI;
    assert_eq!(
        explicit.combine_with_defaults(defaults),
        SanitizerSet::CFI | SanitizerSet::SAFESTACK
    );
}
