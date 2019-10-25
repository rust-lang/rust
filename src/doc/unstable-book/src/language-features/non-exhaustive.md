# `non_exhaustive`

The tracking issue for this feature is: [#44109]

[#44109]: https://github.com/rust-lang/rust/issues/44109

------------------------

The `non_exhaustive` gate allows you to use the `#[non_exhaustive]` attribute
on structs, enums and enum variants. When applied within a crate, users of the
crate will need to use the `_` pattern when matching enums and use the `..`
pattern when matching structs. Enum variants cannot be matched against.
Structs and enum variants marked as `non_exhaustive` will not be able to
be created normally outside of the defining crate. This is demonstrated
below:

```rust,ignore (pseudo-Rust)
use std::error::Error as StdError;

#[non_exhaustive]
pub enum Error {
    Message(String),
    Other,
}
impl StdError for Error {
    fn description(&self) -> &str {
        // This will not error, despite being marked as non_exhaustive, as this
        // enum is defined within the current crate, it can be matched
        // exhaustively.
        match *self {
            Message(ref s) => s,
            Other => "other or unknown error",
        }
    }
}
```

```rust,ignore (pseudo-Rust)
use mycrate::Error;

// This will not error as the non_exhaustive Error enum has been matched with
// a wildcard.
match error {
    Message(ref s) => ...,
    Other => ...,
    _ => ...,
}
```

```rust,ignore (pseudo-Rust)
#[non_exhaustive]
pub struct Config {
    pub window_width: u16,
    pub window_height: u16,
}

// We can create structs as normal within the defining crate when marked as
// non_exhaustive.
let config = Config { window_width: 640, window_height: 480 };

// We can match structs exhaustively when within the defining crate.
if let Ok(Config { window_width, window_height }) = load_config() {
    // ...
}
```

```rust,ignore (pseudo-Rust)
use mycrate::Config;

// We cannot create a struct like normal if it has been marked as
// non_exhaustive.
let config = Config { window_width: 640, window_height: 480 };
// By adding the `..` we can match the config as below outside of the crate
// when marked non_exhaustive.
let &Config { window_width, window_height, .. } = config;
```
