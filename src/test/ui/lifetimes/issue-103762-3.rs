#![feature(anonymous_lifetime_in_impl_trait)]

trait Trait {}

// Pre-1.64 rejected this because of `+ '_`, 1.64 rejects it because of `-> &str`.
// We should keep rejecting it.
fn d1(_: impl Trait + '_)
    -> &str { loop {} } //~ ERROR missing lifetime specifier

// Pre-1.64 rejected these because of `+ '_`, 1.64 allows them with the feature enabled.
// We should keep allowing them.
fn d2(_: &(impl Trait + '_)) -> &str { loop {} }
fn d3(_: &i32, _: impl Trait + '_) -> &str { loop {} }
fn d4(_: impl Trait + '_, _: &i32) -> &str { loop {} }


fn main() {}
