#![deny(rustdoc::invalid_html_tags)]

/// This Vec<32> thing!
// Numbers aren't valid HTML tags, so no error.
pub struct ConstGeneric;

/// This Vec<i32, i32> thing!
// HTML tags cannot contain commas, so no error.
pub struct MultipleGenerics;

/// This <[u32] as Iterator<Item>> thing!
//~^ERROR unclosed HTML tag `Item`
// Some forms of fully-qualified path are simultaneously valid HTML tags
// with attributes. They produce an error, but no suggestion, because figuring
// out if this is valid would require parsing the entire path grammar.
//
// The important part is that we don't produce any *wrong* suggestions.
// While several other examples below are added to make sure we don't
// produce suggestions when given complex paths, this example is the actual
// reason behind not just using the real path parser. It's ambiguous: there's
// no way to locally reason out whether that `[u32]` is intended to be a slice
// or an intra-doc link.
pub struct FullyQualifiedPathsDoNotCount;

/// This <Vec as IntoIter>::Iter thing!
//~^ERROR unclosed HTML tag `Vec`
// Some forms of fully-qualified path are simultaneously valid HTML tags
// with attributes. They produce an error, but no suggestion, because figuring
// out if this is valid would require parsing the entire path grammar.
pub struct FullyQualifiedPathsDoNotCount1;

/// This Vec<Vec as IntoIter>::Iter thing!
//~^ERROR unclosed HTML tag `Vec`
// Some forms of fully-qualified path are simultaneously valid HTML tags
// with attributes. They produce an error, but no suggestion, because figuring
// out if this is valid would require parsing the entire path grammar.
pub struct FullyQualifiedPathsDoNotCount2;

/// This Vec<Vec as IntoIter> thing!
//~^ERROR unclosed HTML tag `Vec`
// Some forms of fully-qualified path are simultaneously valid HTML tags
// with attributes. They produce an error, but no suggestion, because figuring
// out if this is valid would require parsing the entire path grammar.
pub struct FullyQualifiedPathsDoNotCount3;

/// This Vec<Vec<i32> as IntoIter> thing!
//~^ERROR unclosed HTML tag `i32`
// Some forms of fully-qualified path are simultaneously valid HTML tags
// with attributes. They produce an error, but no suggestion, because figuring
// out if this is valid would require parsing the entire path grammar.
pub struct FullyQualifiedPathsDoNotCount4;

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
