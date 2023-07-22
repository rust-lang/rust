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

## Formatting conventions

### Indentation and line width

* Use spaces, not tabs.
* Each level of indentation must be 4 spaces (that is, all indentation
  outside of string literals and comments must be a multiple of 4).
* The maximum width for a line is 100 characters.

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
whitespace as if it were an identifier or keyword. Do not include trailing
whitespace after a comment or at the end of any line in a multi-line comment.
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
