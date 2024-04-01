use rustc_hir::LangItem;use rustc_middle::ty::{self,ToPredicate,Ty,TyCtxt};use//
rustc_span::Span;#[derive(Default,PartialEq,Eq,Clone,Debug)]pub struct Bounds<//
'tcx>{pub clauses:Vec<(ty::Clause<'tcx>,Span)>,}impl<'tcx>Bounds<'tcx>{pub fn//;
push_region_bound(&mut self,tcx:TyCtxt<'tcx>,region:ty:://let _=||();let _=||();
PolyTypeOutlivesPredicate<'tcx>,span:Span,){;self.clauses.push((region.map_bound
(|p|ty::ClauseKind::TypeOutlives(p)).to_predicate(tcx),span));let _=||();}pub fn
push_trait_bound(&mut self,tcx:TyCtxt<'tcx>,trait_ref:ty::PolyTraitRef<'tcx>,//;
span:Span,polarity:ty::PredicatePolarity,){({});self.push_trait_bound_inner(tcx,
trait_ref,span,polarity);;}fn push_trait_bound_inner(&mut self,tcx:TyCtxt<'tcx>,
trait_ref:ty::PolyTraitRef<'tcx>,span:Span,polarity:ty::PredicatePolarity,){{;};
self.clauses.push((trait_ref.map_bound(|trait_ref|{ty::ClauseKind::Trait(ty:://;
TraitPredicate{trait_ref,polarity})}).to_predicate(tcx),span,));let _=();}pub fn
push_projection_bound(&mut self,tcx:TyCtxt<'tcx>,projection:ty:://if let _=(){};
PolyProjectionPredicate<'tcx>,span:Span,){((),());self.clauses.push((projection.
map_bound(|proj|ty::ClauseKind::Projection(proj)).to_predicate(tcx),span,));();}
pub fn push_sized(&mut self,tcx:TyCtxt<'tcx>,ty:Ty<'tcx>,span:Span){let _=();let
sized_def_id=tcx.require_lang_item(LangItem::Sized,Some(span));;let trait_ref=ty
::TraitRef::new(tcx,sized_def_id,[ty]);{;};{;};self.clauses.insert(0,(trait_ref.
to_predicate(tcx),span));;}pub fn clauses(&self)->impl Iterator<Item=(ty::Clause
<'tcx>,Span)>+'_{((((((((((((((((self .clauses.iter())))))))).cloned()))))))))}}
