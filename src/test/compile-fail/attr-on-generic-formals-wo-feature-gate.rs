// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test ensures that attributes on formals in generic parameter
// lists are rejected if feature(generic_param_attrs) is not enabled.
//
// (We are prefixing all tested features with `rustc_`, to ensure that
// the attributes themselves won't be rejected by the compiler when
// using `rustc_attrs` feature. There is a separate compile-fail/ test
// ensuring that the attribute feature-gating works in this context.)

// gate-test-generic_param_attrs

#![feature(rustc_attrs)]
#![allow(dead_code)]

struct StLt<#[rustc_lt_struct] 'a>(&'a u32);
//~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
struct StTy<#[rustc_ty_struct] I>(I);
//~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)

enum EnLt<#[rustc_lt_enum] 'b> { A(&'b u32), B }
//~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
enum EnTy<#[rustc_ty_enum] J> { A(J), B }
//~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)

trait TrLt<#[rustc_lt_trait] 'c> { fn foo(&self, _: &'c [u32]) -> &'c u32; }
//~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
trait TrTy<#[rustc_ty_trait] K> { fn foo(&self, _: K); }
//~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)

type TyLt<#[rustc_lt_type] 'd> = &'d u32;
//~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
type TyTy<#[rustc_ty_type] L> = (L, );
//~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)

impl<#[rustc_lt_inherent] 'e> StLt<'e> { }
//~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
impl<#[rustc_ty_inherent] M> StTy<M> { }
//~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)

impl<#[rustc_lt_impl_for] 'f> TrLt<'f> for StLt<'f> {
    //~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
    fn foo(&self, _: &'f [u32]) -> &'f u32 { loop { } }
}
impl<#[rustc_ty_impl_for] N> TrTy<N> for StTy<N> {
    //~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)
    fn foo(&self, _: N) { }
}

fn f_lt<#[rustc_lt_fn] 'g>(_: &'g [u32]) -> &'g u32 { loop { } }
//~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
fn f_ty<#[rustc_ty_fn] O>(_: O) { }
//~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)

impl<I> StTy<I> {
    fn m_lt<#[rustc_lt_meth] 'h>(_: &'h [u32]) -> &'h u32 { loop { } }
    //~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
    fn m_ty<#[rustc_ty_meth] P>(_: P) { }
    //~^ ERROR attributes on type parameter bindings are experimental (see issue #34761)
}

fn hof_lt<Q>(_: Q)
    where Q: for <#[rustc_lt_hof] 'i> Fn(&'i [u32]) -> &'i u32
    //~^ ERROR attributes on lifetime bindings are experimental (see issue #34761)
{
}

fn main() {

}
