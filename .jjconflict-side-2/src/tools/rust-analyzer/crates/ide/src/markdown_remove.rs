//! Removes markdown from strings.
use pulldown_cmark::{Event, Parser, Tag};

/// Removes all markdown, keeping the text and code blocks
///
/// Currently limited in styling, i.e. no ascii tables or lists
pub(crate) fn remove_markdown(markdown: &str) -> String {
    let mut out = String::new();
    out.reserve_exact(markdown.len());
    let parser = Parser::new(markdown);

    for event in parser {
        match event {
            Event::Text(text) | Event::Code(text) => out.push_str(&text),
            Event::SoftBreak => out.push(' '),
            Event::HardBreak | Event::Rule | Event::End(Tag::CodeBlock(_)) => out.push('\n'),
            Event::End(Tag::Paragraph) => out.push_str("\n\n"),
            Event::Start(_)
            | Event::End(_)
            | Event::Html(_)
            | Event::FootnoteReference(_)
            | Event::TaskListMarker(_) => (),
        }
    }

    if let Some(mut p) = out.rfind(|c| c != '\n') {
        while !out.is_char_boundary(p + 1) {
            p += 1;
        }
        out.drain(p + 1..);
    }

    out
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    #[test]
    fn smoke_test() {
        let res = remove_markdown(
            r##"
A function or function pointer.

Functions are the primary way code is executed within Rust. Function blocks, usually just
called functions, can be defined in a variety of different places and be assigned many
different attributes and modifiers.

Standalone functions that just sit within a module not attached to anything else are common,
but most functions will end up being inside [`impl`] blocks, either on another type itself, or
as a trait impl for that type.

```rust
fn standalone_function() {
    // code
}

pub fn public_thing(argument: bool) -> String {
    // code
    # "".to_string()
}

struct Thing {
    foo: i32,
}

impl Thing {
    pub fn new() -> Self {
        Self {
            foo: 42,
        }
    }
}
```

In addition to presenting fixed types in the form of `fn name(arg: type, ..) -> return_type`,
functions can also declare a list of type parameters along with trait bounds that they fall
into.

```rust
fn generic_function<T: Clone>(x: T) -> (T, T, T) {
    (x.clone(), x.clone(), x.clone())
}

fn generic_where<T>(x: T) -> T
    where T: std::ops::Add<Output = T> + Copy
{
    x + x + x
}
```

Declaring trait bounds in the angle brackets is functionally identical to using a `where`
clause. It's up to the programmer to decide which works better in each situation, but `where`
tends to be better when things get longer than one line.

Along with being made public via `pub`, `fn` can also have an [`extern`] added for use in
FFI.

For more information on the various types of functions and how they're used, consult the [Rust
book] or the [Reference].

[`impl`]: keyword.impl.html
[`extern`]: keyword.extern.html
[Rust book]: ../book/ch03-03-how-functions-work.html
[Reference]: ../reference/items/functions.html
"##,
        );
        expect![[r#"
            A function or function pointer.

            Functions are the primary way code is executed within Rust. Function blocks, usually just called functions, can be defined in a variety of different places and be assigned many different attributes and modifiers.

            Standalone functions that just sit within a module not attached to anything else are common, but most functions will end up being inside impl blocks, either on another type itself, or as a trait impl for that type.

            fn standalone_function() {
                // code
            }

            pub fn public_thing(argument: bool) -> String {
                // code
                # "".to_string()
            }

            struct Thing {
                foo: i32,
            }

            impl Thing {
                pub fn new() -> Self {
                    Self {
                        foo: 42,
                    }
                }
            }

            In addition to presenting fixed types in the form of fn name(arg: type, ..) -> return_type, functions can also declare a list of type parameters along with trait bounds that they fall into.

            fn generic_function<T: Clone>(x: T) -> (T, T, T) {
                (x.clone(), x.clone(), x.clone())
            }

            fn generic_where<T>(x: T) -> T
                where T: std::ops::Add<Output = T> + Copy
            {
                x + x + x
            }

            Declaring trait bounds in the angle brackets is functionally identical to using a where clause. It's up to the programmer to decide which works better in each situation, but where tends to be better when things get longer than one line.

            Along with being made public via pub, fn can also have an extern added for use in FFI.

            For more information on the various types of functions and how they're used, consult the Rust book or the Reference."#]].assert_eq(&res);
    }

    #[test]
    fn on_char_boundary() {
        expect!["a┘"].assert_eq(&remove_markdown("```text\na┘\n```"));
        expect!["وقار"].assert_eq(&remove_markdown("```\nوقار\n```\n"));
    }
}
