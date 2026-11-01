//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// The crate defining an anti-fundamental trait may rely on downstream crates being unable to
// implement it for fundamental types. Without the attribute, the same impls overlap, because
// downstream crates could implement the trait for `FundamentalWrapper<TheirType>`.

#![feature(fundamental)]
#![feature(rustc_attrs)]

#[fundamental]
pub struct FundamentalWrapper<T>(pub T);

#[rustc_anti_fundamental]
pub trait AntiFundamentalTrait {}

pub trait NotAntiFundamentalTrait {}

trait WithAttr {}
impl<T: AntiFundamentalTrait> WithAttr for T {}
impl<T> WithAttr for FundamentalWrapper<T> {}
impl<T> WithAttr for &FundamentalWrapper<T> {}

trait WithoutAttr {}
impl<T: NotAntiFundamentalTrait> WithoutAttr for T {}
impl<T> WithoutAttr for FundamentalWrapper<T> {}
//~^ ERROR conflicting implementations of trait `WithoutAttr`

// Impls the crate itself provides are still taken into account.
#[rustc_anti_fundamental]
pub trait AntiFundamentalWithLocalImpl {}
impl AntiFundamentalWithLocalImpl for FundamentalWrapper<u8> {}

trait OverlapsLocalImpl {}
impl<T: AntiFundamentalWithLocalImpl> OverlapsLocalImpl for T {}
impl<T> OverlapsLocalImpl for FundamentalWrapper<T> {}
//~^ ERROR conflicting implementations of trait `OverlapsLocalImpl`

fn main() {}
