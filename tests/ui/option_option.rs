#![deny(clippy::option_option)]
#![allow(clippy::unnecessary_wraps, clippy::manual_unwrap_or_default)]

const C: Option<Option<i32>> = None;
//~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum if
static S: Option<Option<i32>> = None;
//~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum if

fn input(_: Option<Option<u8>>) {}
//~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum if

fn output() -> Option<Option<u8>> {
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum if
    None
}

fn output_nested() -> Vec<Option<Option<u8>>> {
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum if
    vec![None]
}

// The lint only generates one warning for this
fn output_nested_nested() -> Option<Option<Option<u8>>> {
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum if
    None
}

struct Struct {
    x: Option<Option<u8>>,
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum
}

impl Struct {
    fn struct_fn() -> Option<Option<u8>> {
        //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum
        None
    }
}

trait Trait {
    fn trait_fn() -> Option<Option<u8>>;
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum
}

enum Enum {
    Tuple(Option<Option<u8>>),
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum
    Struct { x: Option<Option<u8>> },
    //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom enum
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
        //~^ ERROR: consider using `Option<T>` instead of `Option<Option<T>>` or a custom
    }

    #[allow(clippy::option_option)]
    fn func<'a, D>(_: D) -> Result<Option<Option<Cow<'a, str>>>, D::Error>
    where
        D: Deserializer<'a>,
    {
        Ok(Some(Some(Cow::Borrowed("hi"))))
    }
}
