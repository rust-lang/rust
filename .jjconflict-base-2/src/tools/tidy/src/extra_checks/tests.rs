use std::str::FromStr;

use crate::extra_checks::{ExtraCheckArg, ExtraCheckKind, ExtraCheckLang, ExtraCheckParseError};

#[test]
fn test_extra_check_arg_from_str_ok() {
    let test_cases = [
        (
            "auto:if-installed:spellcheck",
            ExtraCheckArg {
                auto: true,
                if_installed: true,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            },
        ),
        (
            "if-installed:auto:spellcheck",
            ExtraCheckArg {
                auto: true,
                if_installed: true,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            },
        ),
        (
            "auto:spellcheck",
            ExtraCheckArg {
                auto: true,
                if_installed: false,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            },
        ),
        (
            "if-installed:spellcheck",
            ExtraCheckArg {
                auto: false,
                if_installed: true,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            },
        ),
        (
            "spellcheck",
            ExtraCheckArg {
                auto: false,
                if_installed: false,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            },
        ),
        (
            "js:lint",
            ExtraCheckArg {
                auto: false,
                if_installed: false,
                lang: ExtraCheckLang::Js,
                kind: Some(ExtraCheckKind::Lint),
            },
        ),
    ];

    for (s, expected) in test_cases {
        assert_eq!(ExtraCheckArg::from_str(s), Ok(expected));
    }
}

#[test]
fn test_extra_check_arg_from_str_err() {
    let test_cases = [
        ("some:spellcheck", ExtraCheckParseError::UnknownLang("some".to_string())),
        ("spellcheck:some", ExtraCheckParseError::UnknownKind("some".to_string())),
        ("spellcheck:lint", ExtraCheckParseError::UnsupportedKindForLang),
        ("auto:spellcheck:some", ExtraCheckParseError::UnknownKind("some".to_string())),
        ("auto:js:lint:some", ExtraCheckParseError::TooManyParts),
        ("some", ExtraCheckParseError::UnknownLang("some".to_string())),
        ("auto", ExtraCheckParseError::AutoRequiresLang),
        ("if-installed", ExtraCheckParseError::IfInstalledRequiresLang),
        ("", ExtraCheckParseError::Empty),
    ];

    for (s, expected) in test_cases {
        assert_eq!(ExtraCheckArg::from_str(s), Err(expected));
    }
}
