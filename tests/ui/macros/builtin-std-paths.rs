//@ check-pass

#[derive(
    core::clone::Clone,
    core::marker::Copy,
    core::fmt::Debug,
    core::default::Default,
    core::cmp::Eq,
    core::hash::Hash,
    core::cmp::Ord,
    core::cmp::PartialEq,
    core::cmp::PartialOrd,
)]
struct Core;

#[derive(
    std::clone::Clone,
    std::marker::Copy,
    std::fmt::Debug,
    std::default::Default,
    std::cmp::Eq,
    std::hash::Hash,
    std::cmp::Ord,
    std::cmp::PartialEq,
    std::cmp::PartialOrd,
)]
struct Std;

fn main() {
    core::column!();
    std::column!();
}
