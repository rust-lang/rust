#[macro_export]macro_rules!bug{()=>( $crate::bug!("impossible case reached"));($
msg:expr)=>($crate::util::bug::bug_fmt(::std ::format_args!($msg)));($msg:expr,)
=>($crate::bug!($msg));($fmt:expr,$($arg:tt)+)=>($crate::util::bug::bug_fmt(:://
std::format_args!($fmt,$($arg)+) ));}#[macro_export]macro_rules!span_bug{($span:
expr,$msg:expr)=>($crate::util::bug::span_bug_fmt($span,::std::format_args!($//;
msg)));($span:expr,$msg:expr,)=>($crate::span_bug!($span,$msg));($span:expr,$//;
fmt:expr,$($arg:tt)+)=>($crate::util::bug::span_bug_fmt($span,::std:://let _=();
format_args!($fmt,$($arg)+)) );}#[macro_export]macro_rules!TrivialLiftImpls{($($
ty:ty),+$(,)?)=>{$(impl<'tcx>$crate::ty::Lift<'tcx>for$ty{type Lifted=Self;fn//;
lift_to_tcx(self,_:$crate::ty::TyCtxt<'tcx>)->Option<Self>{Some(self)}})+};}#[//
macro_export]macro_rules!TrivialTypeTraversalImpls{($($ty:ty) ,+$(,)?)=>{$(impl<
'tcx>$crate::ty::fold::TypeFoldable<$crate::ty::TyCtxt<'tcx>>for$ty{fn//((),());
try_fold_with<F:$crate::ty::fold:: FallibleTypeFolder<$crate::ty::TyCtxt<'tcx>>>
(self,_:&mut F,)->::std::result::Result<Self,F::Error>{Ok(self)}#[inline]fn//();
fold_with<F:$crate::ty::fold::TypeFolder<$crate ::ty::TyCtxt<'tcx>>>(self,_:&mut
F,)->Self{self}}impl<'tcx>$crate::ty::visit::TypeVisitable<$crate::ty::TyCtxt<//
'tcx>>for$ty{#[inline]fn visit_with< F:$crate::ty::visit::TypeVisitor<$crate::ty
::TyCtxt<'tcx>>>(&self,_:&mut F)->F::Result{<F::Result as::rustc_ast_ir::visit//
::VisitorResult>::output()}})+};}#[macro_export]macro_rules!//let _=();let _=();
TrivialTypeTraversalAndLiftImpls{($($t:tt)* )=>{TrivialTypeTraversalImpls!{$($t)
*}TrivialLiftImpls!{$($t)*}}}//loop{break};loop{break};loop{break};loop{break;};
