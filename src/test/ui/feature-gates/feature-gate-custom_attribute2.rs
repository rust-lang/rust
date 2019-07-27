// This test ensures that attributes on formals in generic parameter
// lists are included when we are checking for unstable attributes.

// gate-test-custom_attribute

struct StLt<#[lt_struct] 'a>(&'a u32);
//~^ ERROR the attribute `lt_struct` is currently unknown to the compiler
struct StTy<#[ty_struct] I>(I);
//~^ ERROR the attribute `ty_struct` is currently unknown to the compiler

enum EnLt<#[lt_enum] 'b> { A(&'b u32), B }
//~^ ERROR the attribute `lt_enum` is currently unknown to the compiler
enum EnTy<#[ty_enum] J> { A(J), B }
//~^ ERROR the attribute `ty_enum` is currently unknown to the compiler

trait TrLt<#[lt_trait] 'c> { fn foo(&self, _: &'c [u32]) -> &'c u32; }
//~^ ERROR the attribute `lt_trait` is currently unknown to the compiler
trait TrTy<#[ty_trait] K> { fn foo(&self, _: K); }
//~^ ERROR the attribute `ty_trait` is currently unknown to the compiler

type TyLt<#[lt_type] 'd> = &'d u32;
//~^ ERROR the attribute `lt_type` is currently unknown to the compiler
type TyTy<#[ty_type] L> = (L, );
//~^ ERROR the attribute `ty_type` is currently unknown to the compiler

impl<#[lt_inherent] 'e> StLt<'e> { }
//~^ ERROR the attribute `lt_inherent` is currently unknown to the compiler
impl<#[ty_inherent] M> StTy<M> { }
//~^ ERROR the attribute `ty_inherent` is currently unknown to the compiler

impl<#[lt_impl_for] 'f> TrLt<'f> for StLt<'f> {
    //~^ ERROR the attribute `lt_impl_for` is currently unknown to the compiler
    fn foo(&self, _: &'f [u32]) -> &'f u32 { loop { } }
}
impl<#[ty_impl_for] N> TrTy<N> for StTy<N> {
    //~^ ERROR the attribute `ty_impl_for` is currently unknown to the compiler
    fn foo(&self, _: N) { }
}

fn f_lt<#[lt_fn] 'g>(_: &'g [u32]) -> &'g u32 { loop { } }
//~^ ERROR the attribute `lt_fn` is currently unknown to the compiler
fn f_ty<#[ty_fn] O>(_: O) { }
//~^ ERROR the attribute `ty_fn` is currently unknown to the compiler

impl<I> StTy<I> {
    fn m_lt<#[lt_meth] 'h>(_: &'h [u32]) -> &'h u32 { loop { } }
    //~^ ERROR the attribute `lt_meth` is currently unknown to the compiler
    fn m_ty<#[ty_meth] P>(_: P) { }
    //~^ ERROR the attribute `ty_meth` is currently unknown to the compiler
}

fn hof_lt<Q>(_: Q)
    where Q: for <#[lt_hof] 'i> Fn(&'i [u32]) -> &'i u32
    //~^ ERROR the attribute `lt_hof` is currently unknown to the compiler
{
}

fn main() {

}
