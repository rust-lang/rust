# `tool_lints`

The tracking issue for this feature is: [#44690]

[#44690]: https://github.com/rust-lang/rust/issues/44690

------------------------

Tool lints let you use scoped lints, to `allow`, `warn`, `deny` or `forbid` lints of
certain tools.

Currently `clippy` is the only available lint tool.

It is recommended for lint tools to implement the scoped lints like this:

- `#[_(TOOL_NAME::lintname)]`: for lint names
- `#[_(TOOL_NAME::lintgroup)]`: for groups of lints
- `#[_(TOOL_NAME::all)]`: for (almost[^1]) all lints

## An example

```rust
#![feature(tool_lints)]

#![warn(clippy::pedantic)]

#[allow(clippy::filter_map)]
fn main() {
    let v = vec![0; 10];
    let _ = v.into_iter().filter(|&x| x < 1).map(|x| x + 1).collect::<Vec<_>>();
    println!("No filter_map()!");
}
```

[^1]: Some defined lint groups can be excluded here.
