//@ check-pass
// This test is reduced from librustc_ast.  It is just checking that we
// can successfully deal with a "deep" structure, which the drop-check
// was hitting a recursion limit on at one point.


#![allow(non_camel_case_types)]

pub fn noop_fold_impl_item() -> SmallVector<ImplItem> {
    loop  { }
}

pub struct SmallVector<T>(P<T>);
pub struct ImplItem(P<S01_Method>);

struct P<T>(Box<T>);

struct S01_Method(P<S02_Generics>);
struct S02_Generics(P<S03_TyParam>);
struct S03_TyParam(P<S04_TyParamBound>);
struct S04_TyParamBound(S05_PolyTraitRef);
struct S05_PolyTraitRef(S06_TraitRef);
struct S06_TraitRef(S07_Path);
struct S07_Path(Vec<S08_PathSegment>);
struct S08_PathSegment(S09_GenericArgs);
struct S09_GenericArgs(P<S10_ParenthesizedParameterData>);
struct S10_ParenthesizedParameterData(Option<P<S11_Ty>>);
struct S11_Ty(P<S12_Expr>);
struct S12_Expr(P<S13_Block>);
struct S13_Block(Vec<P<S14_Stmt>>);
struct S14_Stmt(P<S15_Decl>);
struct S15_Decl(P<S16_Local>);
struct S16_Local(P<S17_Pat>);
struct S17_Pat(P<S18_Mac>);
struct S18_Mac(Vec<P<S19_TokenTree>>);
struct S19_TokenTree(P<S20_Token>);
struct S20_Token(P<S21_Nonterminal>);
struct S21_Nonterminal(P<S22_Item>);
struct S22_Item(P<S23_EnumDef>);
struct S23_EnumDef(Vec<P<S24_Variant>>);
struct S24_Variant(P<S25_VariantKind>);
struct S25_VariantKind(P<S26_StructDef>);
struct S26_StructDef(Vec<P<S27_StructField>>);
struct S27_StructField(P<S28_StructFieldKind>);
struct S28_StructFieldKind;

pub fn main() {}
