#![deny(rustdoc::invalid_html_tags)]

/// This Vec<32> thing!
// Numbers aren't valid HTML tags, so no error.
pub struct ConstGeneric;

/// This Vec<i32, i32> thing!
// HTML tags cannot contain commas, so no error.
pub struct MultipleGenerics;

/// This Vec<i32 class="test"> thing!
//~^ERROR unclosed HTML tag `i32`
// HTML attributes shouldn't be treated as Rust syntax, so no suggestions.
pub struct TagWithAttributes;

/// This Vec<i32></i32> thing!
// There should be no error, and no suggestion, since the tags are balanced.
pub struct DoNotWarnOnMatchingTags;

/// This Vec</i32> thing!
//~^ERROR unopened HTML tag `i32`
// This should produce an error, but no suggestion.
pub struct EndTagsAreNotValidRustSyntax;

/// This 123<i32> thing!
//~^ERROR unclosed HTML tag `i32`
// This should produce an error, but no suggestion.
pub struct NumbersAreNotPaths;

/// This Vec:<i32> thing!
//~^ERROR unclosed HTML tag `i32`
// This should produce an error, but no suggestion.
pub struct InvalidTurbofish;

/// This [link](https://rust-lang.org)<i32> thing!
//~^ERROR unclosed HTML tag `i32`
// This should produce an error, but no suggestion.
pub struct BareTurbofish;
