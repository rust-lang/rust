//! Regression test for <https://github.com/rust-lang/rust/issues/66805>.

//@ aux-build: nested-macro-rules-definition.rs

extern crate nested_macro_rules_definition;
use nested_macro_rules_definition::*;

macro_rules! make_event_subscription {
    ($(( $field:ident, $ty:ident, $channel:ident )),*) => {
        pub struct EventSubscription($($channel<nested_macro_rules_definition::$ty>::ReaderId),*);
        //~^ ERROR ambiguous associated type
    };
}

all_trigger_fields!(make_event_subscription);
//~^ ERROR macros that expand to items must be delimited with braces or followed by a semicolon

fn main() {}
