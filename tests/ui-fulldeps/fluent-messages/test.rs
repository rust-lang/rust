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

    fluent_messages! { "/definitely_does_not_exist.ftl" }
    //~^ ERROR could not open Fluent resource
}

mod missing_relative {
    use super::fluent_messages;

    fluent_messages! { "../definitely_does_not_exist.ftl" }
    //~^ ERROR could not open Fluent resource
}

mod missing_message {
    use super::fluent_messages;

    fluent_messages! { "./missing-message.ftl" }
    //~^ ERROR could not parse Fluent resource
}

mod duplicate {
    use super::fluent_messages;

    fluent_messages! { "./duplicate.ftl" }
    //~^ ERROR overrides existing message: `no_crate_a_b_key`
}

mod slug_with_hyphens {
    use super::fluent_messages;

    fluent_messages! { "./slug-with-hyphens.ftl" }
    //~^ ERROR name `no_crate_this-slug-has-hyphens` contains a '-' character
}

mod label_with_hyphens {
    use super::fluent_messages;

    fluent_messages! { "./label-with-hyphens.ftl" }
    //~^ ERROR attribute `label-has-hyphens` contains a '-' character
}

mod valid {
    use super::fluent_messages;

    fluent_messages! { "./valid.ftl" }

    mod test_generated {
        use super::{fluent_generated::no_crate_key, DEFAULT_LOCALE_RESOURCE};
    }
}

mod missing_crate_name {
    use super::fluent_messages;

    fluent_messages! { "./missing-crate-name.ftl" }
    //~^ ERROR name `no-crate_foo` contains a '-' character
    //~| ERROR name `with-hyphens` contains a '-' character
    //~| ERROR name `with-hyphens` does not start with the crate name

    mod test_generated {
        use super::{
            fluent_generated::{no_crate_foo, with_hyphens},
            DEFAULT_LOCALE_RESOURCE,
        };
    }
}

mod missing_message_ref {
    use super::fluent_messages;

    fluent_messages! { "./missing-message-ref.ftl" }
    //~^ ERROR referenced message `message` does not exist
}
