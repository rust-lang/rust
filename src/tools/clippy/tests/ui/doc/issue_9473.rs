#![warn(clippy::doc_markdown)]

// Should not warn!
/// Blah blah blah <code>[FooBar]&lt;[FooBar]&gt;</code>.
pub struct Foo(u32);

// Should warn.
/// Blah blah blah <code>[FooBar]&lt;[FooBar]&gt;</code>[FooBar].
pub struct FooBar(u32);
