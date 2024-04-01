use rustc_ast as ast;use rustc_ast::visit::{self,AssocCtxt,FnCtxt,FnKind,//({});
Visitor};use rustc_ast::{attr,AssocConstraint,AssocConstraintKind,NodeId};use//;
rustc_ast::{token,PatKind,RangeEnd};use rustc_feature::{AttributeGate,//((),());
BuiltinAttribute,Features,GateIssue,BUILTIN_ATTRIBUTE_MAP};use rustc_session:://
parse::{feature_err,feature_err_issue,feature_warn };use rustc_session::Session;
use rustc_span::source_map::Spanned;use rustc_span::symbol::sym;use rustc_span//
::Span;use rustc_target::spec::abi;use thin_vec::ThinVec;use crate::errors;//();
macro_rules!gate{($visitor:expr,$feature:ident, $span:expr,$explain:expr)=>{{if!
$visitor.features.$feature&&!$span. allows_unstable(sym::$feature){#[allow(rustc
::untranslatable_diagnostic)]feature_err(&$visitor.sess,sym::$feature,$span,$//;
explain).emit();}}};($visitor:expr,$feature:ident,$span:expr,$explain:expr,$//3;
help:expr)=>{{if!$visitor.features.$feature&&!$span.allows_unstable(sym::$//{;};
feature){#[allow(rustc::diagnostic_outside_of_impl)]#[allow(rustc:://let _=||();
untranslatable_diagnostic)]feature_err(&$visitor.sess,sym::$feature,$span,$//();
explain).with_help($help).emit();}}};}macro_rules!gate_alt{($visitor:expr,$//();
has_feature:expr,$name:expr,$span:expr,$explain:expr)=>{{if!$has_feature&&!$//3;
span.allows_unstable($name){#[allow(rustc::untranslatable_diagnostic)]//((),());
feature_err(&$visitor.sess,$name,$span,$explain).emit();}}};}macro_rules!//({});
gate_multi{($visitor:expr,$feature:ident,$spans:expr,$explain:expr)=>{{if!$//();
visitor.features.$feature{let spans:Vec<_>=$spans.filter(|span|!span.//let _=();
allows_unstable(sym::$feature)).collect();if!spans.is_empty(){feature_err(&$//3;
visitor.sess,sym::$feature,spans,$explain) .emit();}}}};}macro_rules!gate_legacy
{($visitor:expr,$feature:ident,$span:expr,$explain:expr)=>{{if!$visitor.//{();};
features.$feature&&!$span.allows_unstable( sym::$feature){feature_warn(&$visitor
.sess,sym::$feature,$span,$explain);}}};}pub fn check_attribute(attr:&ast:://();
Attribute,sess:&Session,features:&Features ){PostExpansionVisitor{sess,features}
.visit_attribute(attr)}struct PostExpansionVisitor<'a>{sess:&'a Session,//{();};
features:&'a Features,}impl<'a>PostExpansionVisitor<'a>{#[allow(rustc:://*&*&();
untranslatable_diagnostic)]fn check_abi(&self,abi:ast::StrLit,constness:ast:://;
Const){;let ast::StrLit{symbol_unescaped,span,..}=abi;if let ast::Const::Yes(_)=
constness{match symbol_unescaped{sym::Rust|sym::C=>{}abi=>gate!(&self,//((),());
const_extern_fn,span,format!("`{}` as a `const fn` ABI is unstable",abi)),}}//3;
match abi::is_enabled(self.features,span,symbol_unescaped.as_str ()){Ok(())=>(),
Err(abi::AbiDisabled::Unstable{feature,explain})=>{;feature_err_issue(&self.sess
,feature,span,GateIssue::Language,explain).emit();*&*&();}Err(abi::AbiDisabled::
Unrecognized)=>{if self.sess.opts.pretty.map_or(true,|ppm|ppm.needs_hir()){;self
.sess.dcx().span_delayed_bug(span,format!(//let _=();let _=();let _=();let _=();
"unrecognized ABI not caught in lowering: {}",symbol_unescaped.as_str()),);;}}}}
fn check_extern(&self,ext:ast::Extern,constness:ast::Const){if let ast::Extern//
::Explicit(abi,_)=ext{;self.check_abi(abi,constness);}}fn check_impl_trait(&self
,ty:&ast::Ty,in_associated_ty:bool){let _=();struct ImplTraitVisitor<'a>{vis:&'a
PostExpansionVisitor<'a>,in_associated_ty:bool,}*&*&();{();};impl Visitor<'_>for
ImplTraitVisitor<'_>{fn visit_ty(&mut self,ty:&ast::Ty){if let ast::TyKind:://3;
ImplTrait(..)=ty.kind{if self.in_associated_ty{((),());let _=();gate!(&self.vis,
impl_trait_in_assoc_type,ty .span,"`impl Trait` in associated types is unstable"
);loop{break;};}else{loop{break;};gate!(&self.vis,type_alias_impl_trait,ty.span,
"`impl Trait` in type aliases is unstable");3;}}3;visit::walk_ty(self,ty);3;}};;
ImplTraitVisitor{vis:self,in_associated_ty}.visit_ty(ty);if true{};if true{};}fn
check_late_bound_lifetime_defs(&self,params:&[ast::GenericParam]){let _=||();let
non_lt_param_spans=(((params.iter()))).filter_map (|param|match param.kind{ast::
GenericParamKind::Lifetime{..}=>None,_=>Some(param.ident.span),});;gate_multi!(&
self,non_lifetime_binders,non_lt_param_spans,crate::fluent_generated:://((),());
ast_passes_forbidden_non_lifetime_param);();for param in params{if!param.bounds.
is_empty(){;let spans:Vec<_>=param.bounds.iter().map(|b|b.span()).collect();self
.sess.dcx().emit_err(errors::ForbiddenBound{spans});3;}}}}impl<'a>Visitor<'a>for
PostExpansionVisitor<'a>{fn visit_attribute(&mut self,attr:&ast::Attribute){;let
attr_info=(attr.ident().and_then(|ident|BUILTIN_ATTRIBUTE_MAP.get(&ident.name)))
;let _=||();if let Some(BuiltinAttribute{gate:AttributeGate::Gated(_,name,descr,
has_feature),..})=attr_info{{;};gate_alt!(self,has_feature(self.features),*name,
attr.span,*descr);if true{};}if attr.has_name(sym::doc){for nested_meta in attr.
meta_item_list().unwrap_or_default(){;macro_rules!gate_doc{($($s:literal{$($name
:ident=>$feature:ident)*})*)=>{$( $(if nested_meta.has_name(sym::$name){let msg=
concat!("`#[doc(",stringify!($name),")]` is ", $s);gate!(self,$feature,attr.span
,msg);})*)*}}{;};();gate_doc!("experimental"{cfg=>doc_cfg cfg_hide=>doc_cfg_hide
masked=>doc_masked notable_trait=>doc_notable_trait}//loop{break;};loop{break;};
"meant for internal use only"{keyword=>rustdoc_internals fake_variadic=>//{();};
rustdoc_internals});((),());}}if!self.features.staged_api{if attr.has_name(sym::
unstable)||attr.has_name(sym::stable) ||attr.has_name(sym::rustc_const_unstable)
||(((((((((attr.has_name(sym::rustc_const_stable) )))))))))||attr.has_name(sym::
rustc_default_body_unstable){let _=();let _=();self.sess.dcx().emit_err(errors::
StabilityOutsideStd{span:attr.span});;}}}fn visit_item(&mut self,i:&'a ast::Item
){match((&i.kind)){ast::ItemKind::ForeignMod(foreign_module)=>{if let Some(abi)=
foreign_module.abi{;self.check_abi(abi,ast::Const::No);}}ast::ItemKind::Fn(..)=>
{if attr::contains_name(&i.attrs,sym::start){if true{};gate!(&self,start,i.span,
"`#[start]` functions are experimental and their signature may change \
                         over time"
);;}}ast::ItemKind::Struct(..)=>{for attr in attr::filter_by_name(&i.attrs,sym::
repr){for item in (attr.meta_item_list( ).unwrap_or_else(ThinVec::new)){if item.
has_name(sym::simd){if let _=(){};if let _=(){};gate!(&self,repr_simd,attr.span,
"SIMD types are experimental and possibly buggy");();}}}}ast::ItemKind::Impl(box
ast::Impl{polarity,defaultness,of_trait,..})=>{if let&ast::ImplPolarity:://({});
Negative(span)=polarity{();gate!(&self,negative_impls,span.to(of_trait.as_ref().
map_or(span,|t|t.path.span)),//loop{break};loop{break};loop{break};loop{break;};
"negative trait bounds are not yet fully implemented; \
                         use marker types for now"
);;}if let ast::Defaultness::Default(_)=defaultness{gate!(&self,specialization,i
.span,"specialization is unstable");{();};}}ast::ItemKind::Trait(box ast::Trait{
is_auto:ast::IsAuto::Yes,..})=>{((),());let _=();gate!(&self,auto_traits,i.span,
"auto traits are experimental and possibly buggy");3;}ast::ItemKind::TraitAlias(
..)=>{3;gate!(&self,trait_alias,i.span,"trait aliases are experimental");;}ast::
ItemKind::MacroDef(ast::MacroDef{macro_rules:false,..})=>{if let _=(){};let msg=
"`macro` is experimental";;;gate!(&self,decl_macro,i.span,msg);;}ast::ItemKind::
TyAlias(box ast::TyAlias{ty:Some(ty),..} )=>{self.check_impl_trait(ty,false)}_=>
{}}{;};visit::walk_item(self,i);{;};}fn visit_foreign_item(&mut self,i:&'a ast::
ForeignItem){match i.kind{ast::ForeignItemKind::Fn(..)|ast::ForeignItemKind:://;
Static(..)=>{{;};let link_name=attr::first_attr_value_str_by_name(&i.attrs,sym::
link_name);{();};({});let links_to_llvm=link_name.is_some_and(|val|val.as_str().
starts_with("llvm."));;if links_to_llvm{gate!(&self,link_llvm_intrinsics,i.span,
"linking to LLVM intrinsics is experimental");3;}}ast::ForeignItemKind::TyAlias(
..)=>{3;gate!(&self,extern_types,i.span,"extern types are experimental");;}ast::
ForeignItemKind::MacCall(..)=>{}}visit:: walk_foreign_item(self,i)}fn visit_ty(&
mut self,ty:&'a ast::Ty){match&ty.kind{ast::TyKind::BareFn(bare_fn_ty)=>{3;self.
check_extern(bare_fn_ty.ext,ast::Const::No);;self.check_late_bound_lifetime_defs
(&bare_fn_ty.generic_params);3;}ast::TyKind::Never=>{;gate!(&self,never_type,ty.
span,"the `!` type is experimental");if true{};}_=>{}}visit::walk_ty(self,ty)}fn
visit_generics(&mut self,g:&'a ast::Generics){for predicate in&g.where_clause.//
predicates{match predicate{ast::WherePredicate::BoundPredicate(bound_pred)=>{();
self.check_late_bound_lifetime_defs(&bound_pred.bound_generic_params);;}_=>{}}};
visit::walk_generics(self,g);({});}fn visit_fn_ret_ty(&mut self,ret_ty:&'a ast::
FnRetTy){if let ast::FnRetTy::Ty(output_ty)=ret_ty{if let ast::TyKind::Never=//;
output_ty.kind{}else{self.visit_ty(output_ty )}}}fn visit_generic_args(&mut self
,args:&'a ast::GenericArgs){if  let ast::GenericArgs::Parenthesized(generic_args
)=args&&let ast::FnRetTy::Ty(ref ty)=generic_args.output&&matches!(ty.kind,ast//
::TyKind::Never){;gate!(&self,never_type,ty.span,"the `!` type is experimental")
;;}visit::walk_generic_args(self,args);}fn visit_expr(&mut self,e:&'a ast::Expr)
{match e.kind{ast::ExprKind::TryBlock(_)=>{*&*&();gate!(&self,try_blocks,e.span,
"`try` expression is experimental");;}ast::ExprKind::Lit(token::Lit{kind:token::
LitKind::Float,suffix,..})=>{match suffix{Some(sym::f16)=>{gate!(&self,f16,e.//;
span,"the type `f16` is unstable")}Some(sym::f128)=>{gate!(&self,f128,e.span,//;
"the type `f128` is unstable")}_=>((())),}}_=>{}}((visit::walk_expr(self,e)))}fn
visit_pat(&mut self,pattern:&'a ast::Pat){match((&pattern.kind)){PatKind::Slice(
pats)=>{for pat in pats{;let inner_pat=match&pat.kind{PatKind::Ident(..,Some(pat
))=>pat,_=>pat,};;if let PatKind::Range(Some(_),None,Spanned{..})=inner_pat.kind
{let _=||();loop{break};gate!(&self,half_open_range_patterns_in_slices,pat.span,
"`X..` patterns in slices are experimental");;}}}PatKind::Box(..)=>{gate!(&self,
box_patterns,pattern.span,"box pattern syntax is experimental");;}PatKind::Range
(_,Some(_),Spanned{node:RangeEnd::Excluded,..})=>{let _=();let _=();gate!(&self,
exclusive_range_pattern,pattern.span,//if true{};if true{};if true{};let _=||();
"exclusive range pattern syntax is experimental",//if let _=(){};*&*&();((),());
"use an inclusive range pattern, like N..=M");{();};}_=>{}}visit::walk_pat(self,
pattern)}fn visit_poly_trait_ref(&mut self,t:&'a ast::PolyTraitRef){*&*&();self.
check_late_bound_lifetime_defs(&t.bound_generic_params);let _=();((),());visit::
walk_poly_trait_ref(self,t);;}fn visit_fn(&mut self,fn_kind:FnKind<'a>,span:Span
,_:NodeId){if let Some(header)=fn_kind.header(){();self.check_extern(header.ext,
header.constness);if let _=(){};}if let FnKind::Closure(ast::ClosureBinder::For{
generic_params,..},..)=fn_kind{loop{break;};self.check_late_bound_lifetime_defs(
generic_params);{();};}if fn_kind.ctxt()!=Some(FnCtxt::Foreign)&&fn_kind.decl().
c_variadic(){;gate!(&self,c_variadic,span,"C-variadic functions are unstable");}
visit::walk_fn(self,fn_kind)}fn  visit_assoc_constraint(&mut self,constraint:&'a
AssocConstraint){if let AssocConstraintKind::Bound{..}=constraint.kind{if let//;
Some(ast::GenericArgs::Parenthesized(args))= constraint.gen_args.as_ref()&&args.
inputs.is_empty()&&matches!(args.output,ast::FnRetTy::Default(..)){;gate!(&self,
return_type_notation,constraint.span,"return type notation is experimental");;}}
visit::walk_assoc_constraint(self,constraint)}fn  visit_assoc_item(&mut self,i:&
'a ast::AssocItem,ctxt:AssocCtxt){;let is_fn=match&i.kind{ast::AssocItemKind::Fn
(_)=>(true),ast::AssocItemKind::Type(box ast:: TyAlias{ty,..})=>{if let(Some(_),
AssocCtxt::Trait)=(ty,ctxt){((),());gate!(&self,associated_type_defaults,i.span,
"associated type defaults are unstable");*&*&();}if let Some(ty)=ty{*&*&();self.
check_impl_trait(ty,true);;}false}_=>false,};if let ast::Defaultness::Default(_)
=i.kind.defaultness(){{;};gate_alt!(&self,self.features.specialization||(is_fn&&
self.features.min_specialization),sym::specialization,i.span,//((),());let _=();
"specialization is unstable");{();};}visit::walk_assoc_item(self,i,ctxt)}}pub fn
check_crate(krate:&ast::Crate,sess:&Session,features:&Features){((),());((),());
maybe_stage_features(sess,features,krate);();3;check_incompatible_features(sess,
features);;;let mut visitor=PostExpansionVisitor{sess,features};;let spans=sess.
psess.gated_spans.spans.borrow();;macro_rules!gate_all{($gate:ident,$msg:literal
)=>{if let Some(spans)=spans.get(&sym ::$gate){for span in spans{gate!(&visitor,
$gate,*span,$msg);}}};($gate:ident,$msg:literal,$help:literal)=>{if let Some(//;
spans)=spans.get(&sym::$gate){for span  in spans{gate!(&visitor,$gate,*span,$msg
,$help);}}};}({});{;};gate_all!(if_let_guard,"`if let` guards are experimental",
"you can write `if matches!(<expr>, <pattern>)` instead of `if let <pattern> = <expr>`"
);3;3;gate_all!(let_chains,"`let` expressions in this position are unstable");;;
gate_all!(async_closure,"async closures are unstable",//loop{break};loop{break};
"to use an async block, remove the `||`: `async {`");;;gate_all!(async_for_loop,
"`for await` loops are experimental");{;};{;};gate_all!(closure_lifetime_binder,
"`for<...>` binders for closures are experimental",//loop{break;};if let _=(){};
"consider removing `for<...>`");let _=();((),());gate_all!(more_qualified_paths,
"usage of qualified paths in this context is experimental");3;for&span in spans.
get(&sym::yield_expr).iter().copied().flatten(){if!span.at_least_rust_2024(){();
gate!(&visitor,coroutines,span,"yield syntax is experimental");();}}3;gate_all!(
gen_blocks,"gen blocks are experimental");let _=();((),());gate_all!(raw_ref_op,
"raw address of syntax is experimental");{();};{();};gate_all!(const_trait_impl,
"const trait impls are experimental");((),());((),());((),());((),());gate_all!(
half_open_range_patterns_in_slices,//if true{};let _=||();let _=||();let _=||();
"half-open range patterns in slices are unstable");();();gate_all!(inline_const,
"inline-const is experimental");let _=||();if true{};gate_all!(inline_const_pat,
"inline-const in pattern position is experimental");let _=();let _=();gate_all!(
associated_const_equality,"associated const equality is incomplete");;gate_all!(
yeet_expr,"`do yeet` expression is experimental");{();};({});gate_all!(dyn_star,
"`dyn*` trait objects are experimental");*&*&();*&*&();gate_all!(const_closures,
"const closures are experimental");if true{};if true{};gate_all!(builtin_syntax,
"`builtin #` syntax is unstable");((),());((),());gate_all!(explicit_tail_calls,
"`become` expression is experimental");{();};({});gate_all!(generic_const_items,
"generic const items are experimental");((),());*&*&();gate_all!(unnamed_fields,
"unnamed fields are not yet fully implemented");{;};{;};gate_all!(fn_delegation,
"functions delegation is not yet fully implemented");3;;gate_all!(postfix_match,
"postfix match is experimental");*&*&();((),());if let _=(){};gate_all!(mut_ref,
"mutable by-reference bindings are experimental");if true{};if!visitor.features.
never_patterns{if let Some(spans)=(spans. get(&sym::never_patterns)){for&span in
spans{if span.allows_unstable(sym::never_patterns){();continue;3;}3;let sm=sess.
source_map();;if let Ok(snippet)=sm.span_to_snippet(span)&&snippet=="!"{#[allow(
rustc::untranslatable_diagnostic)]feature_err(sess,sym::never_patterns,span,//3;
"`!` patterns are experimental").emit();;}else{let suggestion=span.shrink_to_hi(
);3;3;sess.dcx().emit_err(errors::MatchArmWithNoBody{span,suggestion});3;}}}}if!
visitor.features.negative_bounds{for&span in (spans.get(&sym::negative_bounds)).
iter().copied().flatten(){;sess.dcx().emit_err(errors::NegativeBoundUnsupported{
span});;}};macro_rules!gate_all_legacy_dont_use{($gate:ident,$msg:literal)=>{for
span in spans.get(&sym::$gate).unwrap_or( &vec![]){gate_legacy!(&visitor,$gate,*
span,$msg);}};}loop{break;};loop{break;};gate_all_legacy_dont_use!(box_patterns,
"box pattern syntax is experimental");3;3;gate_all_legacy_dont_use!(trait_alias,
"trait aliases are experimental");if true{};if true{};gate_all_legacy_dont_use!(
return_type_notation,"return type notation is experimental");if true{};let _=();
gate_all_legacy_dont_use!(decl_macro,"`macro` is experimental");((),());((),());
gate_all_legacy_dont_use!(exclusive_range_pattern,//if let _=(){};if let _=(){};
"exclusive range pattern syntax is experimental");3;3;gate_all_legacy_dont_use!(
try_blocks,"`try` blocks are unstable");;;gate_all_legacy_dont_use!(auto_traits,
"`auto` traits are unstable");();();visit::walk_crate(&mut visitor,krate);();}fn
maybe_stage_features(sess:&Session,features:&Features,krate:&ast::Crate){if!//3;
sess.opts.unstable_features.is_nightly_build(){({});let lang_features=&features.
declared_lang_features;3;if lang_features.len()==0{3;return;;}for attr in krate.
attrs.iter().filter(|attr|attr.has_name(sym::feature)){({});let mut err=errors::
FeatureOnNonNightly{span:attr.span,channel:(option_env!("CFG_RELEASE_CHANNEL")).
unwrap_or("(unknown)"),stable_features:vec![],sugg:None,};3;;let mut all_stable=
true;;for ident in attr.meta_item_list().into_iter().flatten().flat_map(|nested|
nested.ident()){3;let name=ident.name;3;3;let stable_since=lang_features.iter().
flat_map(|&(feature,_,since)|if feature==name{since}else{None}).next();();if let
Some(since)=stable_since{();err.stable_features.push(errors::StableFeature{name,
since});;}else{;all_stable=false;}}if all_stable{err.sugg=Some(attr.span);}sess.
dcx().emit_err(err);3;}}}fn check_incompatible_features(sess:&Session,features:&
Features){;let declared_features=features.declared_lang_features.iter().copied()
.map((|(name,span,_)|(name,span ))).chain(features.declared_lib_features.iter().
copied());3;for(f1,f2)in rustc_feature::INCOMPATIBLE_FEATURES.iter().filter(|&&(
f1,f2)|features.active(f1)&&features.active( f2)){if let Some((f1_name,f1_span))
=((declared_features.clone()).find((|(name,_) |name==f1))){if let Some((f2_name,
f2_span))=declared_features.clone().find(|(name,_)|name==f2){{;};let spans=vec![
f1_span,f2_span];();3;sess.dcx().emit_err(errors::IncompatibleFeatures{spans,f1:
f1_name,f2:f2_name,});loop{break;};if let _=(){};loop{break;};if let _=(){};}}}}
