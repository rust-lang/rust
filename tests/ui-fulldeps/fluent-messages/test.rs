//@ normalize-stderr: "could not open Fluent resource:.*" -> "could not open Fluent resource: os-specific message"

#![feature(rustc_private)]
#![crate_type = "lib"]
extern crate rustc_errors;
extern crate rustc_fluent_macro;

/// Copy of the relevant `DiagMessage` variant constructed by `fluent_messages` as it
/// expects `crate::DiagMessage` to exist.
pub enum DiagMessage {
    FluentIdentifier(std::borrow::Cow<'static, str>, Option<std::borrow::Cow<'static, str>>),
}

/// Copy of the relevant `SubdiagMessage` variant constructed by `fluent_messages` as it
/// expects `crate::SubdiagMessage` to exist.
pub enum SubdiagMessage {
    FluentAttr(std::borrow::Cow<'static, str>),
}

mod missing_absolute {
    rustc_fluent_macro::fluent_messages! { "/definitely_does_not_exist.ftl" }
    //~^ ERROR could not open Fluent resource
}

mod missing_relative {
    rustc_fluent_macro::fluent_messages! { "../definitely_does_not_exist.ftl" }
    //~^ ERROR could not open Fluent resource
}

mod missing_message {
    rustc_fluent_macro::fluent_messages! { "./missing-message.ftl" }
    //~^ ERROR could not parse Fluent resource
}

mod duplicate {
    rustc_fluent_macro::fluent_messages! { "./duplicate.ftl" }
    //~^ ERROR overrides existing message: `no_crate_a_b_key`
}

mod slug_with_hyphens {
    rustc_fluent_macro::fluent_messages! { "./slug-with-hyphens.ftl" }
    //~^ ERROR name `no_crate_this-slug-has-hyphens` contains a '-' character
}

mod label_with_hyphens {
    rustc_fluent_macro::fluent_messages! { "./label-with-hyphens.ftl" }
    //~^ ERROR attribute `label-has-hyphens` contains a '-' character
}

mod valid {
    rustc_fluent_macro::fluent_messages! { "./valid.ftl" }

    mod test_generated {
        use super::{fluent_generated::no_crate_key, DEFAULT_LOCALE_RESOURCE};
    }
}

mod missing_crate_name {
    rustc_fluent_macro::fluent_messages! { "./missing-crate-name.ftl" }
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
    rustc_fluent_macro::fluent_messages! { "./missing-message-ref.ftl" }
    //~^ ERROR referenced message `message` does not exist
}

mod bad_escape {
    rustc_fluent_macro::fluent_messages! { "./invalid-escape.ftl" }
    //~^ ERROR invalid escape `\n`
    //~| ERROR invalid escape `\"`
    //~| ERROR invalid escape `\'`
}

mod many_lines {
    rustc_fluent_macro::fluent_messages! { "./many-lines.ftl" }
    //~^ ERROR could not parse Fluent resource
}
