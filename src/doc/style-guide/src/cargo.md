# Cargo.toml conventions

## Formatting conventions

Use the same line width and indentation as Rust code.

Put a blank line between the last key-value pair in a section and the header of
the next section. Do not place a blank line between section headers and the
key-value pairs in that section, or between key-value pairs in a section.

Sort key names alphabetically within each section, with the exception of the
`[package]` section. Put the `[package]` section at the top of the file; put
the `name` and `version` keys in that order at the top of that section,
followed by the remaining keys other than `description` in alphabetical order,
followed by the `description` at the end of that section.

Don't use quotes around any standard key names; use bare keys. Only use quoted
keys for non-standard keys whose names require them, and avoid introducing such
key names when possible.  See the [TOML
specification](https://toml.io/en/v1.0.0#keys) for details.

Put a single space both before and after the `=` between a key and value. Do
not indent any key names; start all key names at the start of a line.

Use multi-line strings (rather than newline escape sequences) for any string
values that include multiple lines, such as the crate description.

For array values, such as a list of authors, put the entire list on the same
line as the key, if it fits. Otherwise, use block indentation: put a newline
after the opening square bracket, indent each item by one indentation level,
put a comma after each item (including the last), and put the closing square
bracket at the start of a line by itself after the last item.

```rust
authors = [
    "A Uthor <a.uthor@example.org>",
    "Another Author <author@example.net>",
]
```

For table values, such as a crate dependency with a path, write the entire
table using curly braces and commas on the same line as the key if it fits. If
the entire table does not fit on the same line as the key, separate it out into
a separate section with key-value pairs:

```toml
[dependencies]
crate1 = { path = "crate1", version = "1.2.3" }

[dependencies.extremely_long_crate_name_goes_here]
path = "extremely_long_path_name_goes_right_here"
version = "4.5.6"
```

## Metadata conventions

The authors list should consist of strings that each contain an author name
followed by an email address in angle brackets: `Full Name <email@address>`.
It should not contain bare email addresses, or names without email addresses.
(The authors list may also include a mailing list address without an associated
name.)

The license field must contain a valid [SPDX
expression](https://spdx.org/spdx-specification-21-web-version#h.jxpfx0ykyb60),
using valid [SPDX license names](https://spdx.org/licenses/). (As an exception,
by widespread convention, the license field may use `/` in place of ` OR `; for
example, `MIT/Apache-2.0`.)

The homepage field, if present, must consist of a single URL, including the
scheme (e.g. `https://example.org/`, not just `example.org`.)

Within the description field, wrap text at 80 columns. Don't start the
description field with the name of the crate (e.g. "cratename is a ..."); just
describe the crate itself. If providing a multi-sentence description, the first
sentence should go on a line by itself and summarize the crate, like the
subject of an email or commit message; subsequent sentences can then describe
the crate in more detail.
