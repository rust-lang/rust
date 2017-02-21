# Comments

Comments in Rust code follow the general C++ style of line (`//`) and
block (`/* ... */`) comment forms. Nested block comments are supported.

Line comments beginning with exactly _three_ slashes (`///`), and block
comments (`/** ... */`), are interpreted as a special syntax for `doc`
[attributes]. That is, they are equivalent to writing
`#[doc="..."]` around the body of the comment, i.e., `/// Foo` turns into
`#[doc="Foo"]`.

Line comments beginning with `//!` and block comments `/*! ... */` are
doc comments that apply to the parent of the comment, rather than the item
that follows.  That is, they are equivalent to writing `#![doc="..."]` around
the body of the comment. `//!` comments are usually used to document
modules that occupy a source file.

Non-doc comments are interpreted as a form of whitespace.

[attributes]: attributes.html
