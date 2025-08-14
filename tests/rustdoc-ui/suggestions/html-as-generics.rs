//@ run-rustfix
#![deny(rustdoc::invalid_html_tags)]

/// This Vec<i32> thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct Generic;

/// This vec::Vec<i32> thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct GenericPath;

/// This i32<i32> thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct PathsCanContainTrailingNumbers;

/// This Vec::<i32> thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct Turbofish;

/// This [link](https://rust-lang.org)::<i32> thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct BareTurbofish;

/// This <span>Vec::<i32></span> thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct Nested;

/// Nested generics Vec<Vec<u32>>
//~^ ERROR unclosed HTML tag `u32`
//~|HELP try marking as source
pub struct NestedGenerics;

/// Generics with path Vec<i32>::Iter
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct GenericsWithPath;

/// Generics with path <Vec<i32>>::Iter
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct NestedGenericsWithPath;

/// Generics with path Vec<Vec<i32>>::Iter
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct NestedGenericsWithPath2;

/// Generics with bump <Vec<i32>>s
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct NestedGenericsWithBump;

/// Generics with bump Vec<Vec<i32>>s
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct NestedGenericsWithBump2;

/// Generics with punct <Vec<i32>>!
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct NestedGenericsWithPunct;

/// Generics with punct Vec<Vec<i32>>!
//~^ ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct NestedGenericsWithPunct2;

/// This [Vec<i32>] thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct IntraDocLink;

/// This [Vec::<i32>] thing!
//~^ERROR unclosed HTML tag `i32`
//~|HELP try marking as source
pub struct IntraDocLinkTurbofish;
