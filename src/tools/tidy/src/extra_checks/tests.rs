use std::str::FromStr;

use crate::extra_checks::{ExtraCheckArg, ExtraCheckKind, ExtraCheckLang, ExtraCheckParseError};

#[test]
fn test_extra_check_arg_from_str_ok() {
    let test_cases = [
        (
            "auto:if-installed:spellcheck",
            Ok(ExtraCheckArg {
                auto: true,
                if_installed: true,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            }),
        ),
        (
            "if-installed:auto:spellcheck",
            Ok(ExtraCheckArg {
                auto: true,
                if_installed: true,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            }),
        ),
        (
            "auto:spellcheck",
            Ok(ExtraCheckArg {
                auto: true,
                if_installed: false,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            }),
        ),
        (
            "if-installed:spellcheck",
            Ok(ExtraCheckArg {
                auto: false,
                if_installed: true,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            }),
        ),
        (
            "spellcheck",
            Ok(ExtraCheckArg {
                auto: false,
                if_installed: false,
                lang: ExtraCheckLang::Spellcheck,
                kind: None,
            }),
        ),
        (
            "js:lint",
            Ok(ExtraCheckArg {
                auto: false,
                if_installed: false,
                lang: ExtraCheckLang::Js,
                kind: Some(ExtraCheckKind::Lint),
            }),
        ),
    ];

    for (s, expected) in test_cases {
        assert_eq!(ExtraCheckArg::from_str(s), expected);
    }
}

#[test]
fn test_extra_check_arg_from_str_err() {
    let test_cases = [
        ("some:spellcheck", Err(ExtraCheckParseError::UnknownLang("some".to_string()))),
        ("spellcheck:some", Err(ExtraCheckParseError::UnknownKind("some".to_string()))),
        ("spellcheck:lint", Err(ExtraCheckParseError::UnsupportedKindForLang)),
        ("auto:spellcheck:some", Err(ExtraCheckParseError::UnknownKind("some".to_string()))),
        ("auto:js:lint:some", Err(ExtraCheckParseError::TooManyParts)),
        ("some", Err(ExtraCheckParseError::UnknownLang("some".to_string()))),
        ("auto", Err(ExtraCheckParseError::AutoRequiresLang)),
        ("if-installed", Err(ExtraCheckParseError::IfInstalledRequiresLang)),
        ("", Err(ExtraCheckParseError::Empty)),
    ];

    for (s, expected) in test_cases {
        assert_eq!(ExtraCheckArg::from_str(s), expected);
    }
}
