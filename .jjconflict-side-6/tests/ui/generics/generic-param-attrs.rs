// This test previously ensured that attributes on formals in generic parameter
// lists are rejected without a feature gate.

//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]

struct StLt<#[rustc_dummy] 'a>(&'a u32);
struct StTy<#[rustc_dummy] I>(I);
enum EnLt<#[rustc_dummy] 'b> { A(&'b u32), B }
enum EnTy<#[rustc_dummy] J> { A(J), B }
trait TrLt<#[rustc_dummy] 'c> { fn foo(&self, _: &'c [u32]) -> &'c u32; }
trait TrTy<#[rustc_dummy] K> { fn foo(&self, _: K); }
type TyLt<#[rustc_dummy] 'd> = &'d u32;
type TyTy<#[rustc_dummy] L> = (L, );

impl<#[rustc_dummy] 'e> StLt<'e> { }
impl<#[rustc_dummy] M> StTy<M> { }
impl<#[rustc_dummy] 'f> TrLt<'f> for StLt<'f> {
    fn foo(&self, _: &'f [u32]) -> &'f u32 { loop { } }
}
impl<#[rustc_dummy] N> TrTy<N> for StTy<N> {
    fn foo(&self, _: N) { }
}

fn f_lt<#[rustc_dummy] 'g>(_: &'g [u32]) -> &'g u32 { loop { } }
fn f_ty<#[rustc_dummy] O>(_: O) { }

impl<I> StTy<I> {
    fn m_lt<#[rustc_dummy] 'h>(_: &'h [u32]) -> &'h u32 { loop { } }
    fn m_ty<#[rustc_dummy] P>(_: P) { }
}

fn hof_lt<Q>(_: Q)
    where Q: for <#[rustc_dummy] 'i> Fn(&'i [u32]) -> &'i u32
{}

fn main() {}
