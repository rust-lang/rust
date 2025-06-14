// Same test as dead-code-module but with 2 doc(test(attr())) at different levels.

//@ edition: 2024
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

#![doc(test(attr(deny(warnings))))]

#[doc(test(attr(allow(dead_code))))]
/// Example
///
/// ```rust,no_run
/// trait OnlyWarning { fn no_deny_warnings(); }
/// ```
static S: u32 = 5;

#[doc(test(attr(allow(dead_code))))]
/// Example
///
/// ```rust,no_run
/// let unused_error = 5;
///
/// fn dead_code_but_no_error() {}
/// ```
const C: u32 = 5;

#[doc(test(attr(allow(dead_code))))]
/// Example
///
/// ```rust,no_run
/// trait OnlyWarningAtA { fn no_deny_warnings(); }
/// ```
struct A {
    #[doc(test(attr(deny(dead_code))))]
    /// Example
    ///
    /// ```rust,no_run
    /// trait DeadCodeInField {}
    /// ```
    field: u32
}

#[doc(test(attr(allow(dead_code))))]
/// Example
///
/// ```rust,no_run
/// trait OnlyWarningAtU { fn no_deny_warnings(); }
/// ```
union U {
    #[doc(test(attr(deny(dead_code))))]
    /// Example
    ///
    /// ```rust,no_run
    /// trait DeadCodeInUnionField {}
    /// ```
    field: u32,
    /// Example
    ///
    /// ```rust,no_run
    /// trait NotDeadCodeInUnionField {}
    /// ```
    field2: u64,
}

#[doc(test(attr(deny(dead_code))))]
/// Example
///
/// ```rust,no_run
/// let not_dead_code_but_unused = 5;
/// ```
enum Enum {
    #[doc(test(attr(allow(dead_code))))]
    /// Example
    ///
    /// ```rust,no_run
    /// trait NotDeadCodeInVariant {}
    ///
    /// fn main() { let unused_in_variant = 5; }
    /// ```
    Variant1,
}

#[doc(test(attr(allow(dead_code))))]
/// Example
///
/// ```rust,no_run
/// trait OnlyWarningAtImplA { fn no_deny_warnings(); }
/// ```
impl A {
    /// Example
    ///
    /// ```rust,no_run
    /// trait NotDeadCodeInImplMethod {}
    /// ```
    fn method() {}
}

#[doc(test(attr(deny(dead_code))))]
/// Example
///
/// ```rust,no_run
/// trait StillDeadCodeAtMyTrait { }
/// ```
trait MyTrait {
    #[doc(test(attr(allow(dead_code))))]
    /// Example
    ///
    /// ```rust,no_run
    /// trait NotDeadCodeAtImplFn {}
    ///
    /// fn main() { let unused_in_impl = 5; }
    /// ```
    fn my_trait_fn();
}
