// This test ensures that attributes on formals in generic parameter
// lists are included when we are checking for unstable attributes.

struct StLt<#[lt_struct] 'a>(&'a u32);
//~^ ERROR cannot find attribute `lt_struct`
struct StTy<#[ty_struct] I>(I);
//~^ ERROR cannot find attribute `ty_struct`

enum EnLt<#[lt_enum] 'b> { A(&'b u32), B }
//~^ ERROR cannot find attribute `lt_enum`
enum EnTy<#[ty_enum] J> { A(J), B }
//~^ ERROR cannot find attribute `ty_enum`

trait TrLt<#[lt_trait] 'c> { fn foo(&self, _: &'c [u32]) -> &'c u32; }
//~^ ERROR cannot find attribute `lt_trait`
trait TrTy<#[ty_trait] K> { fn foo(&self, _: K); }
//~^ ERROR cannot find attribute `ty_trait`

type TyLt<#[lt_type] 'd> = &'d u32;
//~^ ERROR cannot find attribute `lt_type`
type TyTy<#[ty_type] L> = (L, );
//~^ ERROR cannot find attribute `ty_type`

impl<#[lt_inherent] 'e> StLt<'e> { }
//~^ ERROR cannot find attribute `lt_inherent`
impl<#[ty_inherent] M> StTy<M> { }
//~^ ERROR cannot find attribute `ty_inherent`

impl<#[lt_impl_for] 'f> TrLt<'f> for StLt<'f> {
    //~^ ERROR cannot find attribute `lt_impl_for`
    fn foo(&self, _: &'f [u32]) -> &'f u32 { loop { } }
}
impl<#[ty_impl_for] N> TrTy<N> for StTy<N> {
    //~^ ERROR cannot find attribute `ty_impl_for`
    fn foo(&self, _: N) { }
}

fn f_lt<#[lt_fn] 'g>(_: &'g [u32]) -> &'g u32 { loop { } }
//~^ ERROR cannot find attribute `lt_fn`
fn f_ty<#[ty_fn] O>(_: O) { }
//~^ ERROR cannot find attribute `ty_fn`

impl<I> StTy<I> {
    fn m_lt<#[lt_meth] 'h>(_: &'h [u32]) -> &'h u32 { loop { } }
    //~^ ERROR cannot find attribute `lt_meth`
    fn m_ty<#[ty_meth] P>(_: P) { }
    //~^ ERROR cannot find attribute `ty_meth`
}

fn hof_lt<Q>(_: Q)
    where Q: for <#[lt_hof] 'i> Fn(&'i [u32]) -> &'i u32
    //~^ ERROR cannot find attribute `lt_hof`
{
}

fn main() {

}
