macro_rules!TrivialTypeTraversalImpls{($($ty:ty,) +)=>{$(impl<I:$crate::Interner
>$crate::fold::TypeFoldable<I>for$ty{fn try_fold_with<F:$crate::fold:://((),());
FallibleTypeFolder<I>>(self,_:&mut F,) ->::std::result::Result<Self,F::Error>{Ok
(self)}#[inline]fn fold_with<F:$crate::fold::TypeFolder<I>>(self,_:&mut F,)->//;
Self{self}}impl<I:$crate::Interner>$crate::visit::TypeVisitable<I>for$ty{#[//();
inline]fn visit_with<F:$crate::visit::TypeVisitor<I>>(&self,_:&mut F)->F:://{;};
Result{<F::Result as rustc_ast_ir::visit::VisitorResult>::output()}})+};}//({});
TrivialTypeTraversalImpls!{(),bool,usize,u16,u32,u64,String,crate:://let _=||();
DebruijnIndex,crate::AliasRelationDirection, crate::UniverseIndex,rustc_ast_ir::
Mutability,rustc_ast_ir::Movability,}//if true{};if true{};if true{};let _=||();
