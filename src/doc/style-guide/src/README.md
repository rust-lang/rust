# Rust Style Guide

## Motivation - why use a formatting tool?

Formatting code is a mostly mechanical task which takes both time and mental
effort. By using an automatic formatting tool, a programmer is relieved of
this task and can concentrate on more important things.

Furthermore, by sticking to an established style guide (such as this one),
programmers don't need to formulate ad hoc style rules, nor do they need to
debate with other programmers what style rules should be used, saving time,
communication overhead, and mental energy.

Humans comprehend information through pattern matching. By ensuring that all
Rust code has similar formatting, less mental effort is required to comprehend a
new project, lowering the barrier to entry for new developers.

Thus, there are productivity benefits to using a formatting tool (such as
`rustfmt`), and even larger benefits by using a community-consistent
formatting, typically by using a formatting tool's default settings.

## The default Rust style

The Rust Style Guide defines the default Rust style, and *recommends* that
developers and tools follow the default Rust style. Tools such as `rustfmt` use
the style guide as a reference for the default style. Everything in this style
guide, whether or not it uses language such as "must" or the imperative mood
such as "insert a space ..." or "break the line after ...", refers to the
default style.

This should not be interpreted as forbidding developers from following a
non-default style, or forbidding tools from adding any particular configuration
options.

## Bugs

If the style guide differs from rustfmt, that may represent a bug in rustfmt,
or a bug in the style guide; either way, please report it to the style team or
the rustfmt team or both, for investigation and fix.

If implementing a new formatting tool based on the style guide and default Rust
style, please test it on the corpus of existing Rust code, and avoid causing
widespread breakage. The implementation and testing of such a tool may surface
bugs in either the style guide or rustfmt, as well as bugs in the tool itself.

We typically resolve bugs in a fashion that avoids widespread breakage.

## Formatting conventions

### Indentation and line width

- Use spaces, not tabs.
- Each level of indentation must be 4 spaces (that is, all indentation
  outside of string literals and comments must be a multiple of 4).
- The maximum width for a line is 100 characters.

#### Block indent

Prefer block indent over visual indent:

```rust
// Block indent
a_function_call(
    foo,
    bar,
);

// Visual indent
a_function_call(foo,
                bar);
```

This makes for smaller diffs (e.g., if `a_function_call` is renamed in the above
example) and less rightward drift.

### Trailing commas

In comma-separated lists of any kind, use a trailing comma when followed by a
newline:

```rust
function_call(
    argument,
    another_argument,
);

let array = [
    element,
    another_element,
    yet_another_element,
];
```

This makes moving code (e.g., by copy and paste) easier, and makes diffs
smaller, as appending or removing items does not require modifying another line
to add or remove a comma.

### Blank lines

Separate items and statements by either zero or one blank lines (i.e., one or
two newlines). E.g,

```rust
fn foo() {
    let x = ...;

    let y = ...;
    let z = ...;
}

fn bar() {}
fn baz() {}
```

### Trailing whitespace

Do not include trailing whitespace on the end of any line. This includes blank
lines, comment lines, code lines, and string literals.

Note that avoiding trailing whitespace in string literals requires care to
preserve the value of the literal.

### Sorting

In various cases, the default Rust style specifies to sort things. If not
otherwise specified, such sorting should be "version sorting", which ensures
that (for instance) `x8` comes before `x16` even though the character `1` comes
before the character `8`.

For the purposes of the Rust style, to compare two strings for version-sorting:

- Process both strings from beginning to end as two sequences of maximal-length
  chunks, where each chunk consists either of a sequence of characters other
  than ASCII digits, or a sequence of ASCII digits (a numeric chunk), and
  compare corresponding chunks from the strings.
- To compare two numeric chunks, compare them by numeric value, ignoring
  leading zeroes. If the two chunks have equal numeric value, but different
  numbers of leading digits, and this is the first time this has happened for
  these strings, treat the chunks as equal (moving on to the next chunk) but
  remember which string had more leading zeroes.
- To compare two chunks if both are not numeric, compare them by Unicode
  character lexicographically, with two exceptions:
  - `_` (underscore) sorts immediately after ` ` (space) but before any other
    character. (This treats underscore as a word separator, as commonly used in
    identifiers.)
  - Unless otherwise specified, version-sorting should sort non-lowercase
    characters (characters that can start an `UpperCamelCase` identifier)
    before lowercase characters.
- If the comparison reaches the end of the string and considers each pair of
  chunks equal:
  - If one of the numeric comparisons noted the earliest point at which one
    string had more leading zeroes than the other, sort the string with more
    leading zeroes first.
  - Otherwise, the strings are equal.

Note that there exist various algorithms called "version sorting", which
generally try to solve the same problem, but which differ in various ways (such
as in their handling of numbers with leading zeroes). This algorithm
does not purport to precisely match the behavior of any particular other
algorithm, only to produce a simple and satisfying result for Rust formatting.
In particular, this algorithm aims to produce a satisfying result for a set of
symbols that have the same number of leading zeroes, and an acceptable and
easily understandable result for a set of symbols that has varying numbers of
leading zeroes.

As an example, version-sorting will sort the following strings in the order
given:
- `_ZYXW`
- `_abcd`
- `A2`
- `ABCD`
- `Z_YXW`
- `ZY_XW`
- `ZY_XW`
- `ZYXW`
- `ZYXW_`
- `a1`
- `abcd`
- `u_zzz`
- `u8`
- `u16`
- `u32`
- `u64`
- `u128`
- `u256`
- `ua`
- `usize`
- `uz`
- `v000`
- `v00`
- `v0`
- `v0s`
- `v00t`
- `v0u`
- `v001`
- `v01`
- `v1`
- `v009`
- `v09`
- `v9`
- `v010`
- `v10`
- `w005s09t`
- `w5s009t`
- `x64`
- `x86`
- `x86_32`
- `x86_64`
- `x86_128`
- `x87`
- `zyxw`

### [Module-level items](items.md)

### [Statements](statements.md)

### [Expressions](expressions.md)

### [Types](types.md)

### Comments

The following guidelines for comments are recommendations only, a mechanical
formatter might skip formatting of comments.

Prefer line comments (`//`) to block comments (`/* ... */`).

When using line comments, put a single space after the opening sigil.

When using single-line block comments, put a single space after the opening
sigil and before the closing sigil. For multi-line block comments, put a
newline after the opening sigil, and a newline before the closing sigil.

Prefer to put a comment on its own line. Where a comment follows code, put a
single space before it. Where a block comment appears inline, use surrounding
whitespace as if it were an identifier or keyword.

Examples:

```rust
// A comment on an item.
struct Foo { ... }

fn foo() {} // A comment after an item.

pub fn foo(/* a comment before an argument */ x: T) {...}
```

Comments should usually be complete sentences. Start with a capital letter, end
with a period (`.`). An inline block comment may be treated as a note without
punctuation.

Source lines which are entirely a comment should be limited to 80 characters
in length (including comment sigils, but excluding indentation) or the maximum
width of the line (including comment sigils and indentation), whichever is
smaller:

```rust
// This comment goes up to the ................................. 80 char margin.

{
    // This comment is .............................................. 80 chars wide.
}

{
    {
        {
            {
                {
                    {
                        // This comment is limited by the ......................... 100 char margin.
                    }
                }
            }
        }
    }
}
```

#### Doc comments

Prefer line comments (`///`) to block comments (`/** ... */`).

Prefer outer doc comments (`///` or `/** ... */`), only use inner doc comments
(`//!` and `/*! ... */`) to write module-level or crate-level documentation.

Put doc comments before attributes.

### Attributes

Put each attribute on its own line, indented to the level of the item.
In the case of inner attributes (`#!`), indent it to the level of the inside of
the item. Prefer outer attributes, where possible.

For attributes with argument lists, format like functions.

```rust
#[repr(C)]
#[foo(foo, bar)]
#[long_multi_line_attribute(
    split,
    across,
    lines,
)]
struct CRepr {
    #![repr(C)]
    x: f32,
    y: f32,
}
```

For attributes with an equal sign, put a single space before and after the `=`,
e.g., `#[foo = 42]`.

There must only be a single `derive` attribute. Note for tool authors: if
combining multiple `derive` attributes into a single attribute, the ordering of
the derived names must generally be preserved for correctness:
`#[derive(Foo)] #[derive(Bar)] struct Baz;` must be formatted to
`#[derive(Foo, Bar)] struct Baz;`.

### *small* items

In many places in this guide we specify formatting that depends on a code
construct being *small*. For example, single-line vs multi-line struct
literals:

```rust
// Normal formatting
Foo {
    f1: an_expression,
    f2: another_expression(),
}

// "small" formatting
Foo { f1, f2 }
```

We leave it to individual tools to decide on exactly what *small* means. In
particular, tools are free to use different definitions in different
circumstances.

Some suitable heuristics are the size of the item (in characters) or the
complexity of an item (for example, that all components must be simple names,
not more complex sub-expressions). For more discussion on suitable heuristics,
see [this issue](https://github.com/rust-lang-nursery/fmt-rfcs/issues/47).

## [Non-formatting conventions](advice.md)

## [Cargo.toml conventions](cargo.md)

## [Principles used for deciding these guidelines](principles.md)
