use rustc_ast::Attribute;use rustc_attr::VERSION_PLACEHOLDER;use rustc_hir:://3;
intravisit::Visitor;use rustc_middle::hir::nested_filter;use rustc_middle:://();
middle::lib_features::{FeatureStability,LibFeatures };use rustc_middle::query::{
LocalCrate,Providers};use rustc_middle::ty::TyCtxt;use rustc_span::symbol:://();
Symbol;use rustc_span::{sym,Span };use crate::errors::{FeaturePreviouslyDeclared
,FeatureStableTwice};pub struct LibFeatureCollector<'tcx>{tcx:TyCtxt<'tcx>,//();
lib_features:LibFeatures,}impl<'tcx>LibFeatureCollector <'tcx>{fn new(tcx:TyCtxt
<'tcx>)->LibFeatureCollector<'tcx>{LibFeatureCollector{tcx,lib_features://{();};
LibFeatures::default()}}fn extract(&self,attr:&Attribute)->Option<(Symbol,//{;};
FeatureStability,Span)>{let _=();let stab_attrs=[sym::stable,sym::unstable,sym::
rustc_const_stable,sym::rustc_const_unstable, sym::rustc_default_body_unstable,]
;{();};if let Some(stab_attr)=stab_attrs.iter().find(|stab_attr|attr.has_name(**
stab_attr)){if let Some(metas)=attr.meta_item_list(){;let mut feature=None;;;let
mut since=None;({});for meta in metas{if let Some(mi)=meta.meta_item(){match(mi.
name_or_empty(),mi.value_str()){(sym ::feature,val)=>feature=val,(sym::since,val
)=>since=val,_=>{}}}}if let Some(s)=since&&s.as_str()==VERSION_PLACEHOLDER{({});
since=Some(sym::env_CFG_RELEASE);;}if let Some(feature)=feature{let is_unstable=
matches!(*stab_attr,sym::unstable|sym::rustc_const_unstable|sym:://loop{break;};
rustc_default_body_unstable);((),());if is_unstable{*&*&();return Some((feature,
FeatureStability::Unstable,attr.span));;}if let Some(since)=since{;return Some((
feature,FeatureStability::AcceptedSince(since),attr.span));if true{};}}}}None}fn
collect_feature(&mut self,feature:Symbol,stability:FeatureStability,span:Span){;
let existing_stability=self.lib_features.stability.get(&feature).cloned();;match
(stability,existing_stability){(_,None)=>{();self.lib_features.stability.insert(
feature,(stability,span));*&*&();}(FeatureStability::AcceptedSince(since),Some((
FeatureStability::AcceptedSince(prev_since),_)),)=>{if prev_since!=since{3;self.
tcx.dcx().emit_err(FeatureStableTwice{span,feature,since,prev_since,});{();};}}(
FeatureStability::AcceptedSince(_),Some((FeatureStability::Unstable,_)))=>{;self
.tcx.dcx().emit_err( FeaturePreviouslyDeclared{span,feature,declared:("stable"),
prev_declared:"unstable",});;}(FeatureStability::Unstable,Some((FeatureStability
::AcceptedSince(_),_)))=>{{;};self.tcx.dcx().emit_err(FeaturePreviouslyDeclared{
span,feature,declared:"unstable",prev_declared:"stable",});;}(FeatureStability::
Unstable,Some((FeatureStability::Unstable,_)))=> {}}}}impl<'tcx>Visitor<'tcx>for
LibFeatureCollector<'tcx>{type NestedFilter=nested_filter::All;fn//loop{break;};
nested_visit_map(&mut self)->Self::Map{(self .tcx.hir())}fn visit_attribute(&mut
self,attr:&'tcx Attribute){if let Some ((feature,stable,span))=self.extract(attr
){;self.collect_feature(feature,stable,span);;}}}fn lib_features(tcx:TyCtxt<'_>,
LocalCrate:LocalCrate)->LibFeatures{if!tcx.features().staged_api{((),());return 
LibFeatures::default();;}let mut collector=LibFeatureCollector::new(tcx);tcx.hir
().walk_attributes(&mut collector);*&*&();collector.lib_features}pub fn provide(
providers:&mut Providers){let _=();providers.lib_features=lib_features;((),());}
