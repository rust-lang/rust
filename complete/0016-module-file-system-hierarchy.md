- Start Date: 2014-05-02
- RFC PR: [rust-lang/rfcs#63](https://github.com/rust-lang/rfcs/pull/63)
- Rust Issue: [rust-lang/rust#14180](https://github.com/rust-lang/rust/issues/14180)

# Summary

The rules about the places `mod foo;` can be used are tightened to only permit
its use in a crate root and in `mod.rs` files, to ensure a more sane
correspondence between module structure and file system hierarchy. Most
notably, this prevents a common newbie error where a module is loaded multiple
times, leading to surprising incompatibility between them. This proposal does
not take away one's ability to shoot oneself in the foot should one really
desire to; it just removes almost all of the rope, leaving only mixed
metaphors.

# Motivation

It is a common newbie mistake to write things like this:

`lib.rs`:

```rust
mod foo;
pub mod bar;
```

`foo.rs`:

```rust
mod baz;

pub fn foo(_baz: baz::Baz) { }
```

`bar.rs`:

```rust
mod baz;
use foo::foo;

pub fn bar(baz: baz::Baz) {
    foo(baz)
}
```

`baz.rs`:

```rust
pub struct Baz;
```

This fails to compile because `foo::foo()` wants a `foo::baz::Baz`, while
`bar::bar()` is giving it a `bar::baz::Baz`.

Such a situation, importing one file multiple times, is exceedingly rarely what
the user actually wanted to do, but the present design allows it to occur
without warning the user. The alterations contained herein ensure that there is
no situation where such double loading can occur without deliberate intent via
`#[path = "….rs"]`.

# Drawbacks

None known.

# Detailed design

When a `mod foo;` statement is used, the compiler attempts to find a suitable
file. At present, it just blindly seeks for `foo.rs` or `foo/mod.rs` (relative
to the file under parsing).

The new behaviour will only permit `mod foo;` if at least one of the following
conditions hold:

- The file under parsing is the crate root, or

- The file under parsing is a `mod.rs`, or

- `#[path]` is specified, e.g. `#[path = "foo.rs"] mod foo;`.

In layman's terms, the file under parsing must "own" the directory, so to
speak.

# Alternatives

The rationale is covered in the summary. This is the simplest repair to the
current lack of structure; all alternatives would be more complex and invasive.

One non-invasive alternative is a lint which would detect double loads. This is
less desirable than the solution discussed in this RFC as it doesn't fix the
underlying problem which can, fortunately, be fairly easily fixed.

# Unresolved questions

None.
