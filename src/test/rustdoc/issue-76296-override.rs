// ignore-tidy-linelength
// aux-build: issue-76296-1.rs
// aux-build: issue-76296-2.rs
// compile-flags: -Z unstable-options
// compile-flags: --edition 2018
// compile-flags: --extern priv:foo1={{build-base}}/issue-76296-override/auxiliary/libfoo-1.so
// compile-flags: --extern priv:foo2={{build-base}}/issue-76296-override/auxiliary/libfoo-2.so
// compile-flags: --extern-html-root-url {{build-base}}/issue-76296-override/auxiliary/libfoo-1.so=https://example.com/override/v1
// compile-flags: --extern-html-root-url {{build-base}}/issue-76296-override/auxiliary/libfoo-2.so=https://example.com/override/v2

// @has 'issue_76296_override/index.html'
// @matches - '//a[@href="https://example.com/override/v1/foo/struct.Foo1.html"]' '^Foo1$'
// @matches - '//a[@href="https://example.com/override/v2/foo/struct.Foo2.html"]' '^Foo2$'

#[doc(no_inline)]
pub use foo1::Foo1;

#[doc(no_inline)]
pub use foo2::Foo2;

// @has 'issue_76296_override/fn.foo1.html'
// @matches - '//a[@href="https://example.com/override/v1/foo/struct.Foo1.html"]' '^foo1::Foo1$'
// @matches - '//a[@href="https://example.com/override/v1/foo/struct.Foo1.html"]' '^Foo1$'

/// Makes a [`foo1::Foo1`]
pub fn foo1() -> Foo1 { foo1::Foo1 }

// @has 'issue_76296_override/fn.foo2.html'
// @matches - '//a[@href="https://example.com/override/v2/foo/struct.Foo2.html"]' '^foo2::Foo2$'
// @matches - '//a[@href="https://example.com/override/v2/foo/struct.Foo2.html"]' '^Foo2$'

/// Makes a [`foo2::Foo2`]
pub fn foo2() -> Foo2 { foo2::Foo2 }
