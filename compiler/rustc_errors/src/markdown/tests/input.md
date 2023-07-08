# H1 Heading [with a link][remote-link]

H1 content: **some words in bold** and `so does inline code`

## H2 Heading

H2 content: _some words in italic_

### H3 Heading

H3 content: ~~strikethrough~~ text

#### H4 Heading

H4 content: A [simple link](https://docs.rs) and a [remote-link].

---

A section break was above. We can also do paragraph breaks:

(new paragraph) and unordered lists:

- Item 1 in `code`
- Item 2 in _italics_

Or ordered:

1. Item 1 in **bold**
2. Item 2 with some long lines that should wrap: Lorem ipsum dolor sit amet,
   consectetur adipiscing elit. Aenean ac mattis nunc. Phasellus elit quam,
   pulvinar ac risus in, dictum vehicula turpis. Vestibulum neque est, accumsan
   in cursus sit amet, dictum a nunc. Suspendisse aliquet, lorem eu eleifend
   accumsan, magna neque sodales nisi, a aliquet lectus leo eu sem.

---

## Code

Both `inline code` and code blocks are supported:

```rust
/// A rust enum
#[derive(Debug, PartialEq, Clone)]
enum Foo {
    /// Start of line
    Bar
}
```

[remote-link]: http://docs.rs
