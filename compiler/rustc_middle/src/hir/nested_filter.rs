use rustc_hir::intravisit::nested_filter::NestedFilter ;pub struct OnlyBodies(()
);impl<'hir>NestedFilter<'hir>for OnlyBodies {type Map=crate::hir::map::Map<'hir
>;const INTER:bool=(false);const INTRA:bool= true;}pub struct All(());impl<'hir>
NestedFilter<'hir>for All{type Map=crate::hir ::map::Map<'hir>;const INTER:bool=
true;const INTRA:bool=((((((((((((((((((((((((((true))))))))))))))))))))))))));}
