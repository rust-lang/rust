#![deny(clippy::option_option)]
#![allow(clippy::unnecessary_wraps)]

fn input(_: Option<Option<u8>>) {}

fn output() -> Option<Option<u8>> {
    None
}

fn output_nested() -> Vec<Option<Option<u8>>> {
    vec![None]
}

// The lint only generates one warning for this
fn output_nested_nested() -> Option<Option<Option<u8>>> {
    None
}

struct Struct {
    x: Option<Option<u8>>,
}

impl Struct {
    fn struct_fn() -> Option<Option<u8>> {
        None
    }
}

trait Trait {
    fn trait_fn() -> Option<Option<u8>>;
}

enum Enum {
    Tuple(Option<Option<u8>>),
    Struct { x: Option<Option<u8>> },
}

// The lint allows this
type OptionOption = Option<Option<u32>>;

// The lint allows this
fn output_type_alias() -> OptionOption {
    None
}

// The line allows this
impl Trait for Struct {
    fn trait_fn() -> Option<Option<u8>> {
        None
    }
}

fn main() {
    input(None);
    output();
    output_nested();

    // The lint allows this
    let local: Option<Option<u8>> = None;

    // The lint allows this
    let expr = Some(Some(true));
}

extern crate serde;
mod issue_4298 {
    use serde::{Deserialize, Deserializer, Serialize};
    use std::borrow::Cow;

    #[derive(Serialize, Deserialize)]
    struct Foo<'a> {
        #[serde(deserialize_with = "func")]
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        #[serde(borrow)]
        foo: Option<Option<Cow<'a, str>>>,
    }

    #[allow(clippy::option_option)]
    fn func<'a, D>(_: D) -> Result<Option<Option<Cow<'a, str>>>, D::Error>
    where
        D: Deserializer<'a>,
    {
        Ok(Some(Some(Cow::Borrowed("hi"))))
    }
}
