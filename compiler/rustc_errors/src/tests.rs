use std::sync::{Arc, LazyLock};

use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_error_messages::fluent_bundle::resolver::errors::{ReferenceKind, ResolverError};
use rustc_error_messages::{DiagMessage, langid};

use crate::FluentBundle;
use crate::error::{TranslateError, TranslateErrorKind};
use crate::fluent_bundle::*;
use crate::translation::Translator;

fn make_translator(ftl: &'static str) -> Translator {
    let resource = FluentResource::try_new(ftl.into()).expect("Failed to parse an FTL string.");

    let langid_en = langid!("en-US");

    let mut bundle: FluentBundle =
        IntoDynSyncSend(crate::fluent_bundle::bundle::FluentBundle::new_concurrent(vec![
            langid_en,
        ]));

    bundle.add_resource(resource).expect("Failed to add FTL resources to the bundle.");

    Translator {
        fluent_bundle: None,
        fallback_fluent_bundle: Arc::new(LazyLock::new(Box::new(|| bundle))),
    }
}

#[test]
fn wellformed_fluent() {
    let translator = make_translator("mir_build_borrow_of_moved_value = borrow of moved value
    .label = value moved into `{$name}` here
    .occurs_because_label = move occurs because `{$name}` has type `{$ty}` which does not implement the `Copy` trait
    .value_borrowed_label = value borrowed here after move
    .suggestion = borrow this binding in the pattern to avoid moving the value");

    let mut args = FluentArgs::new();
    args.set("name", "Foo");
    args.set("ty", "std::string::String");
    {
        let message = DiagMessage::FluentIdentifier(
            "mir_build_borrow_of_moved_value".into(),
            Some("suggestion".into()),
        );

        assert_eq!(
            translator.translate_message(&message, &args).unwrap(),
            "borrow this binding in the pattern to avoid moving the value"
        );
    }

    {
        let message = DiagMessage::FluentIdentifier(
            "mir_build_borrow_of_moved_value".into(),
            Some("value_borrowed_label".into()),
        );

        assert_eq!(
            translator.translate_message(&message, &args).unwrap(),
            "value borrowed here after move"
        );
    }

    {
        let message = DiagMessage::FluentIdentifier(
            "mir_build_borrow_of_moved_value".into(),
            Some("occurs_because_label".into()),
        );

        assert_eq!(
            translator.translate_message(&message, &args).unwrap(),
            "move occurs because `\u{2068}Foo\u{2069}` has type `\u{2068}std::string::String\u{2069}` which does not implement the `Copy` trait"
        );

        {
            let message = DiagMessage::FluentIdentifier(
                "mir_build_borrow_of_moved_value".into(),
                Some("label".into()),
            );

            assert_eq!(
                translator.translate_message(&message, &args).unwrap(),
                "value moved into `\u{2068}Foo\u{2069}` here"
            );
        }
    }
}

#[test]
fn misformed_fluent() {
    let translator = make_translator("mir_build_borrow_of_moved_value = borrow of moved value
    .label = value moved into `{name}` here
    .occurs_because_label = move occurs because `{$oops}` has type `{$ty}` which does not implement the `Copy` trait
    .suggestion = borrow this binding in the pattern to avoid moving the value");

    let mut args = FluentArgs::new();
    args.set("name", "Foo");
    args.set("ty", "std::string::String");
    {
        let message = DiagMessage::FluentIdentifier(
            "mir_build_borrow_of_moved_value".into(),
            Some("value_borrowed_label".into()),
        );

        let err = translator.translate_message(&message, &args).unwrap_err();
        assert!(
            matches!(
                &err,
                TranslateError::Two {
                    primary: box TranslateError::One {
                        kind: TranslateErrorKind::PrimaryBundleMissing,
                        ..
                    },
                    fallback: box TranslateError::One {
                        kind: TranslateErrorKind::AttributeMissing { attr: "value_borrowed_label" },
                        ..
                    }
                }
            ),
            "{err:#?}"
        );
        assert_eq!(
            format!("{err}"),
            "failed while formatting fluent string `mir_build_borrow_of_moved_value`: \nthe attribute `value_borrowed_label` was missing\nhelp: add `.value_borrowed_label = <message>`\n"
        );
    }

    {
        let message = DiagMessage::FluentIdentifier(
            "mir_build_borrow_of_moved_value".into(),
            Some("label".into()),
        );

        let err = translator.translate_message(&message, &args).unwrap_err();
        if let TranslateError::Two {
            primary: box TranslateError::One { kind: TranslateErrorKind::PrimaryBundleMissing, .. },
            fallback: box TranslateError::One { kind: TranslateErrorKind::Fluent { errs }, .. },
        } = &err
            && let [
                FluentError::ResolverError(ResolverError::Reference(
                    ReferenceKind::Message { id, .. } | ReferenceKind::Variable { id, .. },
                )),
            ] = &**errs
            && id == "name"
        {
        } else {
            panic!("{err:#?}")
        };
        assert_eq!(
            format!("{err}"),
            "failed while formatting fluent string `mir_build_borrow_of_moved_value`: \nargument `name` exists but was not referenced correctly\nhelp: try using `{$name}` instead\n"
        );
    }

    {
        let message = DiagMessage::FluentIdentifier(
            "mir_build_borrow_of_moved_value".into(),
            Some("occurs_because_label".into()),
        );

        let err = translator.translate_message(&message, &args).unwrap_err();
        if let TranslateError::Two {
            primary: box TranslateError::One { kind: TranslateErrorKind::PrimaryBundleMissing, .. },
            fallback: box TranslateError::One { kind: TranslateErrorKind::Fluent { errs }, .. },
        } = &err
            && let [
                FluentError::ResolverError(ResolverError::Reference(
                    ReferenceKind::Message { id, .. } | ReferenceKind::Variable { id, .. },
                )),
            ] = &**errs
            && id == "oops"
        {
        } else {
            panic!("{err:#?}")
        };
        assert_eq!(
            format!("{err}"),
            "failed while formatting fluent string `mir_build_borrow_of_moved_value`: \nthe fluent string has an argument `oops` that was not found.\nhelp: the arguments `name` and `ty` are available\n"
        );
    }
}
