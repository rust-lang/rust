// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test ensures we can attach attributes to the formals in all
// places where generic parameter lists occur, assuming appropriate
// feature gates are enabled.
//
// (We are prefixing all tested features with `rustc_`, to ensure that
// the attributes themselves won't be rejected by the compiler when
// using `rustc_attrs` feature. There is a separate compile-fail/ test
// ensuring that the attribute feature-gating works in this context.)

#![feature(generic_param_attrs, rustc_attrs)]
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
{
}

fn main() {

}
