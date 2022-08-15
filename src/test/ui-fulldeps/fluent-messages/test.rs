// normalize-stderr-test "note.*" -> "note: os-specific message"

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_macros;
use rustc_macros::fluent_messages;

/// Copy of the relevant `DiagnosticMessage` variant constructed by `fluent_messages` as it
/// expects `crate::DiagnosticMessage` to exist.
pub enum DiagnosticMessage {
    FluentIdentifier(std::borrow::Cow<'static, str>, Option<std::borrow::Cow<'static, str>>),
}

/// Copy of the relevant `SubdiagnosticMessage` variant constructed by `fluent_messages` as it
/// expects `crate::SubdiagnosticMessage` to exist.
pub enum SubdiagnosticMessage {
    FluentAttr(std::borrow::Cow<'static, str>),
}

mod missing_absolute {
    use super::fluent_messages;

    fluent_messages! {
        missing_absolute => "/definitely_does_not_exist.ftl",
//~^ ERROR could not open Fluent resource
    }
}

mod missing_relative {
    use super::fluent_messages;

    fluent_messages! {
        missing_relative => "../definitely_does_not_exist.ftl",
//~^ ERROR could not open Fluent resource
    }
}

mod missing_message {
    use super::fluent_messages;

    fluent_messages! {
        missing_message => "./missing-message.ftl",
//~^ ERROR could not parse Fluent resource
    }
}

mod duplicate {
    use super::fluent_messages;

    fluent_messages! {
        a => "./duplicate-a.ftl",
        b => "./duplicate-b.ftl",
//~^ ERROR overrides existing message: `key`
    }
}

mod slug_with_hyphens {
    use super::fluent_messages;

    fluent_messages! {
        slug_with_hyphens => "./slug-with-hyphens.ftl",
//~^ ERROR name `this-slug-has-hyphens` contains a '-' character
    }
}

mod label_with_hyphens {
    use super::fluent_messages;

    fluent_messages! {
        label_with_hyphens => "./label-with-hyphens.ftl",
//~^ ERROR attribute `label-has-hyphens` contains a '-' character
    }
}

mod valid {
    use super::fluent_messages;

    fluent_messages! {
        valid => "./valid.ftl",
    }

    use self::fluent_generated::{DEFAULT_LOCALE_RESOURCES, valid::valid};
}
