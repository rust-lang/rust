// This test previously ensured that attributes on formals in generic parameter
// lists are rejected without a feature gate.
//
// (We are prefixing all tested features with `rustc_`, to ensure that
// the attributes themselves won't be rejected by the compiler when
// using `rustc_attrs` feature. There is a separate compile-fail/ test
// ensuring that the attribute feature-gating works in this context.)

// build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]
#![allow(dead_code)]

struct StLt<#[rustc_lt_struct] 'a>(&'a u32);
struct StTy<#[rustc_ty_struct] I>(I);
enum EnLt<#[rustc_lt_enum] 'b> { A(&'b u32), B }
enum EnTy<#[rustc_ty_enum] J> { A(J), B }
trait TrLt<#[rustc_lt_trait] 'c> { fn foo(&self, _: &'c [u32]) -> &'c u32; }
trait TrTy<#[rustc_ty_trait] K> { fn foo(&self, _: K); }
type TyLt<#[rustc_lt_type] 'd> = &'d u32;
type TyTy<#[rustc_ty_type] L> = (L, );

impl<#[rustc_lt_inherent] 'e> StLt<'e> { }
impl<#[rustc_ty_inherent] M> StTy<M> { }
impl<#[rustc_lt_impl_for] 'f> TrLt<'f> for StLt<'f> {
    fn foo(&self, _: &'f [u32]) -> &'f u32 { loop { } }
}
impl<#[rustc_ty_impl_for] N> TrTy<N> for StTy<N> {
    fn foo(&self, _: N) { }
}

fn f_lt<#[rustc_lt_fn] 'g>(_: &'g [u32]) -> &'g u32 { loop { } }
fn f_ty<#[rustc_ty_fn] O>(_: O) { }

impl<I> StTy<I> {
    fn m_lt<#[rustc_lt_meth] 'h>(_: &'h [u32]) -> &'h u32 { loop { } }
    fn m_ty<#[rustc_ty_meth] P>(_: P) { }
}

fn hof_lt<Q>(_: Q)
    where Q: for <#[rustc_lt_hof] 'i> Fn(&'i [u32]) -> &'i u32
{}

fn main() {}
