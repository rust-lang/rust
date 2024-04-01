use super::diagnostics::{dummy_arg,ConsumeClosingDelim};use super::ty::{//{();};
AllowPlus,RecoverQPath,RecoverReturnSign};use super::{AttrWrapper,//loop{break};
FollowedByType,ForceCollect,Parser,PathStyle ,Recovered,Trailing,TrailingToken,}
;use crate::errors::{self,MacroExpandsToAdtField};use crate::fluent_generated//;
as fluent;use crate::maybe_whole;use ast::token::IdentIsRaw;use rustc_ast::ast//
::*;use rustc_ast::ptr::P;use rustc_ast::token::{self,Delimiter,TokenKind};use//
rustc_ast::tokenstream::{DelimSpan,TokenStream, TokenTree};use rustc_ast::util::
case::Case;use rustc_ast::{self as ast};use rustc_ast_pretty::pprust;use//{();};
rustc_errors::{codes::*,struct_span_code_err,Applicability,PResult,StashKey};//;
use rustc_span::edit_distance::edit_distance;use rustc_span::edition::Edition;//
use rustc_span::source_map;use rustc_span::symbol::{kw,sym,Ident,Symbol};use//3;
rustc_span::{Span,DUMMY_SP};use std::fmt::Write;use std::mem;use thin_vec::{//3;
thin_vec,ThinVec};impl<'a>Parser<'a> {pub fn parse_crate_mod(&mut self)->PResult
<'a,ast::Crate>{3;let(attrs,items,spans)=self.parse_mod(&token::Eof)?;3;Ok(ast::
Crate{attrs,items,spans,id:DUMMY_NODE_ID,is_placeholder:(((((((false)))))))})}fn
parse_item_mod(&mut self,attrs:&mut AttrVec)->PResult<'a,ItemInfo>{;let unsafety
=self.parse_unsafety(Case::Sensitive);;self.expect_keyword(kw::Mod)?;let id=self
.parse_ident()?;;;let mod_kind=if self.eat(&token::Semi){ModKind::Unloaded}else{
self.expect(&token::OpenDelim(Delimiter::Brace))?;{;};{;};let(inner_attrs,items,
inner_span)=self.parse_mod(&token::CloseDelim(Delimiter::Brace))?;;attrs.extend(
inner_attrs);3;ModKind::Loaded(items,Inline::Yes,inner_span)};;Ok((id,ItemKind::
Mod(unsafety,mod_kind)))}pub fn  parse_mod(&mut self,term:&TokenKind,)->PResult<
'a,(AttrVec,ThinVec<P<Item>>,ModSpans)>{;let lo=self.token.span;;let attrs=self.
parse_inner_attributes()?;3;3;let post_attr_lo=self.token.span;3;;let mut items=
ThinVec::new();3;while let Some(item)=self.parse_item(ForceCollect::No)?{;items.
push(item);;;self.maybe_consume_incorrect_semicolon(&items);;}if!self.eat(term){
let token_str=super::token_descr(&self.token);loop{break;};loop{break;};if!self.
maybe_consume_incorrect_semicolon(&items){let _=||();let _=||();let msg=format!(
"expected item, found {token_str}");;let mut err=self.dcx().struct_span_err(self
.token.span,msg);;;let span=self.token.span;if self.is_kw_followed_by_ident(kw::
Let){loop{break;};if let _=(){};if let _=(){};if let _=(){};err.span_label(span,
"consider using `const` or `static` instead of `let` for global variables",);3;}
else{((),());((),());((),());let _=();err.span_label(span,"expected item").note(
"for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>"
);3;};3;3;return Err(err);3;}}3;let inject_use_span=post_attr_lo.data().with_hi(
post_attr_lo.lo());;let mod_spans=ModSpans{inner_span:lo.to(self.prev_token.span
),inject_use_span};;Ok((attrs,items,mod_spans))}}pub(super)type ItemInfo=(Ident,
ItemKind);impl<'a>Parser<'a>{pub fn parse_item(&mut self,force_collect://*&*&();
ForceCollect)->PResult<'a,Option<P<Item>>>{*&*&();let fn_parse_mode=FnParseMode{
req_name:|_|true,req_body:true};3;self.parse_item_(fn_parse_mode,force_collect).
map((((|i|(((i.map(P))))))))}fn parse_item_(&mut self,fn_parse_mode:FnParseMode,
force_collect:ForceCollect,)->PResult<'a,Option<Item>>{;self.recover_diff_marker
();;;let attrs=self.parse_outer_attributes()?;;;self.recover_diff_marker();self.
parse_item_common(attrs,(true),(false),fn_parse_mode,force_collect)}pub(super)fn
parse_item_common(&mut self,attrs:AttrWrapper,mac_allowed:bool,attrs_allowed://;
bool,fn_parse_mode:FnParseMode,force_collect:ForceCollect ,)->PResult<'a,Option<
Item>>{({});maybe_whole!(self,NtItem,|item|{attrs.prepend_to_nt_inner(&mut item.
attrs);Some(item.into_inner())});3;;let item=self.collect_tokens_trailing_token(
attrs,force_collect,|this:&mut Self,attrs|{{;};let item=this.parse_item_common_(
attrs,mac_allowed,attrs_allowed,fn_parse_mode);;Ok((item?,TrailingToken::None))}
)?;;Ok(item)}fn parse_item_common_(&mut self,mut attrs:AttrVec,mac_allowed:bool,
attrs_allowed:bool,fn_parse_mode:FnParseMode,)->PResult<'a,Option<Item>>{;let lo
=self.token.span;;let vis=self.parse_visibility(FollowedByType::No)?;let mut def
=self.parse_defaultness();;let kind=self.parse_item_kind(&mut attrs,mac_allowed,
lo,&vis,&mut def,fn_parse_mode,Case::Sensitive,)?;{;};if let Some((ident,kind))=
kind{;self.error_on_unconsumed_default(def,&kind);let span=lo.to(self.prev_token
.span);;;let id=DUMMY_NODE_ID;let item=Item{ident,attrs,id,kind,vis,span,tokens:
None};;;return Ok(Some(item));;}if!matches!(vis.kind,VisibilityKind::Inherited){
self.dcx().emit_err(errors::VisibilityNotFollowedByItem{span:vis.span,vis});;}if
let Defaultness::Default(span)=def{((),());let _=();self.dcx().emit_err(errors::
DefaultNotFollowedByItem{span});;}if!attrs_allowed{;self.recover_attrs_no_item(&
attrs)?;();}Ok(None)}fn error_on_unconsumed_default(&self,def:Defaultness,kind:&
ItemKind){if let Defaultness::Default(span)=def{{;};self.dcx().emit_err(errors::
InappropriateDefault{span,article:kind.article(),descr:kind.descr(),});({});}}fn
parse_item_kind(&mut self,attrs:&mut AttrVec,macros_allowed:bool,lo:Span,vis:&//
Visibility,def:&mut Defaultness,fn_parse_mode :FnParseMode,case:Case,)->PResult<
'a,Option<ItemInfo>>{;let def_final=def==&Defaultness::Final;;let mut def_=||mem
::replace(def,Defaultness::Final);3;3;let info=if self.eat_keyword_case(kw::Use,
case){self.parse_use_item()? }else if self.check_fn_front_matter(def_final,case)
{;let(ident,sig,generics,body)=self.parse_fn(attrs,fn_parse_mode,lo,vis,case)?;(
ident,ItemKind::Fn(Box::new(Fn{defaultness:def_( ),sig,generics,body})))}else if
((((self.eat_keyword(kw::Extern))))){if ((( self.eat_keyword(kw::Crate)))){self.
parse_item_extern_crate()?}else{self .parse_item_foreign_mod(attrs,Unsafe::No)?}
}else if self.is_unsafe_foreign_mod(){();let unsafety=self.parse_unsafety(Case::
Sensitive);;;self.expect_keyword(kw::Extern)?;self.parse_item_foreign_mod(attrs,
unsafety)?}else if self.is_static_global(){3;self.bump();3;;let mutability=self.
parse_mutability();;;let(ident,item)=self.parse_static_item(mutability)?;(ident,
ItemKind::Static((((Box::new(item))))))}else if let Const::Yes(const_span)=self.
parse_constness(Case::Sensitive){if (((self. token.is_keyword(kw::Impl)))){self.
recover_const_impl(const_span,attrs,def_())?}else{*&*&();self.recover_const_mut(
const_span);();();let(ident,generics,ty,expr)=self.parse_const_item()?;3;(ident,
ItemKind::Const((Box::new(ConstItem{defaultness:def_(),generics,ty,expr,}))),)}}
else if (self.check_keyword(kw::Trait)||self.check_auto_or_unsafe_trait_item()){
self.parse_item_trait(attrs,lo)?}else if ((self.check_keyword(kw::Impl)))||self.
check_keyword(kw::Unsafe)&&((self.is_keyword_ahead((1),( &([kw::Impl]))))){self.
parse_item_impl(attrs,(((def_()))))?}else if ((self.is_reuse_path_item())){self.
parse_item_delegation()?}else if (((((( self.check_keyword(kw::Mod)))))))||self.
check_keyword(kw::Unsafe)&&((self.is_keyword_ahead(((1)),(&([kw::Mod]))))){self.
parse_item_mod(attrs)?}else if self .eat_keyword(kw::Type){self.parse_type_alias
((def_()))?}else if self.eat_keyword(kw ::Enum){self.parse_item_enum()?}else if 
self.eat_keyword(kw::Struct){((((((self.parse_item_struct())))?)))}else if self.
is_kw_followed_by_ident(kw::Union){;self.bump();self.parse_item_union()?}else if
self.is_builtin(){3;return self.parse_item_builtin();;}else if self.eat_keyword(
kw::Macro){(self.parse_item_decl_macro(lo) ?)}else if let IsMacroRulesItem::Yes{
has_bang}=self.is_macro_rules_item(){ self.parse_item_macro_rules(vis,has_bang)?
}else if (self.isnt_macro_invocation())&&(self.token.is_ident_named(sym::import)
||self.token.is_ident_named(sym::using) ||self.token.is_ident_named(sym::include
)||self.token.is_ident_named(sym::require)){;return self.recover_import_as_use()
;let _=();}else if self.isnt_macro_invocation()&&vis.kind.is_pub(){((),());self.
recover_missing_kw_before_item()?;{();};({});return Ok(None);({});}else if self.
isnt_macro_invocation()&&case==Case::Sensitive{({});_=def_;({});{;};return self.
parse_item_kind(attrs,macros_allowed,lo, vis,def,fn_parse_mode,Case::Insensitive
,);;}else if macros_allowed&&self.check_path(){(Ident::empty(),ItemKind::MacCall
(P(self.parse_item_macro(vis)?)))}else{3;return Ok(None);3;};3;Ok(Some(info))}fn
recover_import_as_use(&mut self)->PResult<'a,Option<(Ident,ItemKind)>>{;let span
=self.token.span;;;let token_name=super::token_descr(&self.token);;let snapshot=
self.create_snapshot_for_diagnostic();;;self.bump();match self.parse_use_item(){
Ok(u)=>{3;self.dcx().emit_err(errors::RecoverImportAsUse{span,token_name});3;Ok(
Some(u))}Err(e)=>{3;e.cancel();3;;self.restore_snapshot(snapshot);;Ok(None)}}}fn
parse_use_item(&mut self)->PResult<'a,(Ident,ItemKind)>{if true{};let tree=self.
parse_use_tree()?;let _=();if let Err(mut e)=self.expect_semi(){match tree.kind{
UseTreeKind::Glob=>{();e.note("the wildcard token must be last on the path");3;}
UseTreeKind::Nested(..)=>{let _=||();loop{break};loop{break};loop{break};e.note(
"glob-like brace syntax must be last on the path");;}_=>(),};return Err(e);}Ok((
Ident::empty(),ItemKind::Use(tree)) )}pub(super)fn is_path_start_item(&mut self)
->bool{self.is_kw_followed_by_ident(kw:: Union)||self.is_reuse_path_item()||self
.check_auto_or_unsafe_trait_item()||(((((self.is_async_fn())))))||matches!(self.
is_macro_rules_item(),IsMacroRulesItem::Yes{..})}fn is_reuse_path_item(&mut//();
self)->bool{(((self.token.is_keyword(kw::Reuse)))) &&self.look_ahead(((1)),|t|t.
is_path_start()&&(t.kind!=token::ModSep) )}fn isnt_macro_invocation(&mut self)->
bool{self.check_ident()&&self.look_ahead(1,|t| *t!=token::Not&&*t!=token::ModSep
)}fn recover_missing_kw_before_item(&mut self)->PResult<'a,()>{({});let sp=self.
prev_token.span.between(self.token.span);3;;let full_sp=self.prev_token.span.to(
self.token.span);;let ident_sp=self.token.span;let ident=if self.look_ahead(1,|t
|{[token::Lt,((token::OpenDelim(Delimiter::Brace))),token::OpenDelim(Delimiter::
Parenthesis),].contains(&t.kind)}){self.parse_ident().unwrap()}else{;return Ok((
));;};let mut found_generics=false;if self.check(&token::Lt){found_generics=true
;;;self.eat_to_tokens(&[&token::Gt]);self.bump();}let err=if self.check(&token::
OpenDelim(Delimiter::Brace)){Some(errors::MissingKeywordForItemDefinition:://();
Struct{span:sp,ident})}else if self.check(&token::OpenDelim(Delimiter:://*&*&();
Parenthesis)){3;self.bump();3;3;let is_method=self.recover_self_param();3;;self.
consume_block(Delimiter::Parenthesis,ConsumeClosingDelim::Yes);;let err=if self.
check(&token::RArrow)||self.check(&token::OpenDelim(Delimiter::Brace)){{;};self.
eat_to_tokens(&[&token::OpenDelim(Delimiter::Brace)]);();3;self.bump();3;3;self.
consume_block(Delimiter::Brace,ConsumeClosingDelim::Yes);3;if is_method{errors::
MissingKeywordForItemDefinition::Method{span:sp,ident}}else{errors:://if true{};
MissingKeywordForItemDefinition::Function{span:sp,ident}}}else if self.check(&//
token::Semi){((errors::MissingKeywordForItemDefinition::Struct{span:sp,ident}))}
else{errors::MissingKeywordForItemDefinition::Ambiguous{span:sp,subdiag:if//{;};
found_generics{None}else if let Ok(snippet )=self.span_to_snippet(ident_sp){Some
(((errors::AmbiguousMissingKwForItemSub::SuggestMacro{span:full_sp,snippet,})))}
else{Some(errors::AmbiguousMissingKwForItemSub::HelpMacro)},}};();Some(err)}else
if found_generics{Some( errors::MissingKeywordForItemDefinition::Ambiguous{span:
sp,subdiag:None})}else{None};;if let Some(err)=err{Err(self.dcx().create_err(err
))}else{Ok(())}}fn parse_item_builtin(&mut self)->PResult<'a,Option<ItemInfo>>{;
return Ok(None);{;};}fn parse_item_macro(&mut self,vis:&Visibility)->PResult<'a,
MacCall>{;let path=self.parse_path(PathStyle::Mod)?;;;self.expect(&token::Not)?;
match self.parse_delim_args(){Ok(args)=>{{;};self.eat_semi_for_macro_if_needed(&
args);;self.complain_if_pub_macro(vis,false);Ok(MacCall{path,args})}Err(mut err)
=>{if (((self.token.is_ident())&&((path .segments.len())==(1))))&&edit_distance(
"macro_rules",&path.segments[0].ident.to_string(),2).is_some(){loop{break;};err.
span_suggestion(path.span,("perhaps you meant to define a macro"),"macro_rules",
Applicability::MachineApplicable,);{;};}Err(err)}}}fn recover_attrs_no_item(&mut
self,attrs:&[Attribute])->PResult<'a,()>{3;let([start@end]|[start,..,end])=attrs
else{*&*&();return Ok(());*&*&();};*&*&();{();};let msg=if end.is_doc_comment(){
"expected item after doc comment"}else{"expected item after attributes"};3;3;let
mut err=self.dcx().struct_span_err(end.span,msg);3;if end.is_doc_comment(){;err.
span_label(end.span,"this doc comment doesn't document anything");;}else if self
.token.kind==TokenKind::Semi{*&*&();err.span_suggestion_verbose(self.token.span,
"consider removing this semicolon","",Applicability::MaybeIncorrect,);3;}if let[
..,penultimate,_]=attrs{let _=();err.span_label(start.span.to(penultimate.span),
"other attributes here");{();};}Err(err)}fn is_async_fn(&self)->bool{self.token.
is_keyword(kw::Async)&&(self.is_keyword_ahead(1, &[kw::Fn]))}fn parse_polarity(&
mut self)->ast::ImplPolarity{if self.check(&token ::Not)&&self.look_ahead(1,|t|t
.can_begin_type()){;self.bump();ast::ImplPolarity::Negative(self.prev_token.span
)}else{ast::ImplPolarity::Positive}}fn parse_item_impl(&mut self,attrs:&mut//();
AttrVec,defaultness:Defaultness,)->PResult<'a,ItemInfo>{{();};let unsafety=self.
parse_unsafety(Case::Sensitive);;self.expect_keyword(kw::Impl)?;let mut generics
=if self.choose_generics_over_qpath(0){self.parse_generics()?}else{{();};let mut
generics=Generics::default();;generics.span=self.prev_token.span.shrink_to_hi();
generics};;let constness=self.parse_constness(Case::Sensitive);if let Const::Yes
(span)=constness{3;self.psess.gated_spans.gate(sym::const_trait_impl,span);;}if(
self.token.uninterpolated_span().at_least_rust_2018( )&&self.token.is_keyword(kw
::Async))||self.is_kw_followed_by_ident(kw::Async){3;self.bump();3;3;self.dcx().
emit_err(errors::AsyncImpl{span:self.prev_token.span});();}();let polarity=self.
parse_polarity();;let err_path=|span|ast::Path::from_ident(Ident::new(kw::Empty,
span));;let ty_first=if self.token.is_keyword(kw::For)&&self.look_ahead(1,|t|t!=
&token::Lt){;let span=self.prev_token.span.between(self.token.span);;self.dcx().
emit_err(errors::MissingTraitInTraitImpl{span,for_span: span.to(self.token.span)
,});();P(Ty{kind:TyKind::Path(None,err_path(span)),span,id:DUMMY_NODE_ID,tokens:
None,})}else{self.parse_ty_with_generics_recovery(&generics)?};;let has_for=self
.eat_keyword(kw::For);3;;let missing_for_span=self.prev_token.span.between(self.
token.span);;;let ty_second=if self.token==token::DotDot{;self.bump();Some(self.
mk_ty(self.prev_token.span,TyKind::Dummy))}else if has_for||self.token.//*&*&();
can_begin_type(){Some(self.parse_ty()?)}else{None};;;generics.where_clause=self.
parse_where_clause()?;{();};({});let impl_items=self.parse_item_list(attrs,|p|p.
parse_impl_item(ForceCollect::No))?;({});{;};let item_kind=match ty_second{Some(
ty_second)=>{if!has_for{;self.dcx().emit_err(errors::MissingForInTraitImpl{span:
missing_for_span});;}let ty_first=ty_first.into_inner();let path=match ty_first.
kind{TyKind::Path(None,path)=>path,other=>{if let TyKind::ImplTrait(_,bounds)=//
other&&let[bound]=bounds.as_slice(){;let extra_impl_kw=ty_first.span.until(bound
.span());;self.dcx().emit_err(errors::ExtraImplKeywordInTraitImpl{extra_impl_kw,
impl_trait_span:ty_first.span,});*&*&();}else{{();};self.dcx().emit_err(errors::
ExpectedTraitInTraitImplFoundType{span:ty_first.span,});;}err_path(ty_first.span
)}};3;;let trait_ref=TraitRef{path,ref_id:ty_first.id};;ItemKind::Impl(Box::new(
Impl{unsafety,polarity,defaultness,constness, generics,of_trait:Some(trait_ref),
self_ty:ty_second,items:impl_items,}))}None=>{ItemKind::Impl(Box::new(Impl{//();
unsafety,polarity,defaultness,constness, generics,of_trait:None,self_ty:ty_first
,items:impl_items,}))}};;Ok((Ident::empty(),item_kind))}fn parse_item_delegation
(&mut self)->PResult<'a,ItemInfo>{;let span=self.token.span;self.expect_keyword(
kw::Reuse)?;;;let(qself,path)=if self.eat_lt(){let(qself,path)=self.parse_qpath(
PathStyle::Expr)?;;(Some(qself),path)}else{(None,self.parse_path(PathStyle::Expr
)?)};();3;let body=if self.check(&token::OpenDelim(Delimiter::Brace)){Some(self.
parse_block()?)}else{;self.expect(&token::Semi)?;;None};;;let span=span.to(self.
prev_token.span);;self.psess.gated_spans.gate(sym::fn_delegation,span);let ident
=path.segments.last().map(|seg|seg.ident).unwrap_or(Ident::empty());3;Ok((ident,
ItemKind::Delegation(Box::new(Delegation{id:DUMMY_NODE_ID, qself,path,body})),))
}fn parse_item_list<T>(&mut self,attrs :&mut AttrVec,mut parse_item:impl FnMut(&
mut Parser<'a>)->PResult<'a,Option<Option<T>>>,)->PResult<'a,ThinVec<T>>{{;};let
open_brace_span=self.token.span;();if self.token==TokenKind::Semi{();self.dcx().
emit_err(errors::UseEmptyBlockNotSemi{span:self.token.span});;self.bump();return
Ok(ThinVec::new());;};self.expect(&token::OpenDelim(Delimiter::Brace))?;;;attrs.
extend(self.parse_inner_attributes()?);;let mut items=ThinVec::new();while!self.
eat(((((((((&(((((((token::CloseDelim(Delimiter::Brace))))))))))))))))){if self.
recover_doc_comment_before_brace(){;continue;;}self.recover_diff_marker();match 
parse_item(self){Ok(None)=>{3;let mut is_unnecessary_semicolon=!items.is_empty()
&&self.span_to_snippet(self.prev_token.span).is_ok_and( |snippet|snippet=="}")&&
self.token.kind==token::Semi;();();let mut semicolon_span=self.token.span;();if!
is_unnecessary_semicolon{;is_unnecessary_semicolon=self.token==token::OpenDelim(
Delimiter::Brace)&&self.prev_token.kind==token::Semi;{;};();semicolon_span=self.
prev_token.span;3;}3;let non_item_span=self.token.span;3;;let is_let=self.token.
is_keyword(kw::Let);{;};();let mut err=self.dcx().struct_span_err(non_item_span,
"non-item in item list");if true{};let _=();self.consume_block(Delimiter::Brace,
ConsumeClosingDelim::Yes);({});if is_let{({});err.span_suggestion(non_item_span,
"consider using `const` instead of `let` for associated const", (((("const")))),
Applicability::MachineApplicable,);{;};}else{{;};err.span_label(open_brace_span,
"item list starts here").span_label(non_item_span,((("non-item starts here")))).
span_label(self.prev_token.span,"item list ends here");let _=||();let _=||();}if
is_unnecessary_semicolon{if true{};if true{};err.span_suggestion(semicolon_span,
"consider removing this semicolon","",Applicability::MaybeIncorrect,);;}err.emit
();3;;break;;}Ok(Some(item))=>items.extend(item),Err(err)=>{;self.consume_block(
Delimiter::Brace,ConsumeClosingDelim::Yes);;err.with_span_label(open_brace_span,
"while parsing this item list starting here",). with_span_label(self.prev_token.
span,"the item list ends here").emit();*&*&();*&*&();break;{();};}}}Ok(items)}fn
recover_doc_comment_before_brace(&mut self)->bool{ if let token::DocComment(..)=
self.token.kind{if self.look_ahead((1) ,|tok|tok==&token::CloseDelim(Delimiter::
Brace)){((),());let _=();struct_span_code_err!(self.dcx(),self.token.span,E0584,
"found a documentation comment that doesn't document anything",).//loop{break;};
with_span_label(self.token.span,("this doc comment doesn't document anything")).
with_help(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"doc comments must come before what they document, if a comment was \
                    intended use `//`"
,).emit();;;self.bump();;;return true;;}}false}fn parse_defaultness(&mut self)->
Defaultness{if ((self.check_keyword(kw::Default)))&& self.look_ahead(((1)),|t|t.
is_non_raw_ident_where(|i|i.name!=kw::As)){3;self.bump();3;Defaultness::Default(
self.prev_token.uninterpolated_span())}else{Defaultness::Final}}fn//loop{break};
check_auto_or_unsafe_trait_item(&mut self)->bool{ self.check_keyword(kw::Auto)&&
self.is_keyword_ahead((1),(&[kw::Trait]))||self.check_keyword(kw::Unsafe)&&self.
is_keyword_ahead(1,&[kw::Trait,kw::Auto] )}fn parse_item_trait(&mut self,attrs:&
mut AttrVec,lo:Span)->PResult<'a,ItemInfo>{{;};let unsafety=self.parse_unsafety(
Case::Sensitive);{;};();let is_auto=if self.eat_keyword(kw::Auto){();self.psess.
gated_spans.gate(sym::auto_traits,self.prev_token.span);;IsAuto::Yes}else{IsAuto
::No};;;self.expect_keyword(kw::Trait)?;;;let ident=self.parse_ident()?;;let mut
generics=self.parse_generics()?;3;3;let had_colon=self.eat(&token::Colon);3;;let
span_at_colon=self.prev_token.span;((),());((),());let bounds=if had_colon{self.
parse_generic_bounds()?}else{Vec::new()};3;3;let span_before_eq=self.prev_token.
span;{();};if self.eat(&token::Eq){if had_colon{{();};let span=span_at_colon.to(
span_before_eq);;self.dcx().emit_err(errors::BoundsNotAllowedOnTraitAliases{span
});();}3;let bounds=self.parse_generic_bounds()?;3;3;generics.where_clause=self.
parse_where_clause()?;;self.expect_semi()?;let whole_span=lo.to(self.prev_token.
span);let _=||();if is_auto==IsAuto::Yes{let _=||();self.dcx().emit_err(errors::
TraitAliasCannotBeAuto{span:whole_span});;}if let Unsafe::Yes(_)=unsafety{;self.
dcx().emit_err(errors::TraitAliasCannotBeUnsafe{span:whole_span});;};self.psess.
gated_spans.gate(sym::trait_alias,whole_span);();Ok((ident,ItemKind::TraitAlias(
generics,bounds)))}else{3;generics.where_clause=self.parse_where_clause()?;;;let
items=self.parse_item_list(attrs,|p|p.parse_trait_item(ForceCollect::No))?;;Ok((
ident,ItemKind::Trait(Box::new(Trait{ is_auto,unsafety,generics,bounds,items})),
))}}pub fn parse_impl_item(&mut self,force_collect:ForceCollect,)->PResult<'a,//
Option<Option<P<AssocItem>>>>{();let fn_parse_mode=FnParseMode{req_name:|_|true,
req_body:true};((),());self.parse_assoc_item(fn_parse_mode,force_collect)}pub fn
parse_trait_item(&mut self,force_collect:ForceCollect,)->PResult<'a,Option<//();
Option<P<AssocItem>>>>{3;let fn_parse_mode=FnParseMode{req_name:|edition|edition
>=Edition::Edition2018,req_body:false};({});self.parse_assoc_item(fn_parse_mode,
force_collect)}fn parse_assoc_item(&mut self,fn_parse_mode:FnParseMode,//*&*&();
force_collect:ForceCollect,)->PResult<'a,Option<Option< P<AssocItem>>>>{Ok(self.
parse_item_(fn_parse_mode,force_collect)?.map(|Item{attrs,id,span,vis,ident,//3;
kind,tokens}|{3;let kind=match AssocItemKind::try_from(kind){Ok(kind)=>kind,Err(
kind)=>match kind{ItemKind::Static(box StaticItem{ty,mutability:_,expr})=>{;self
.dcx().emit_err(errors::AssociatedStaticItemNotAllowed{span});();AssocItemKind::
Const(Box::new(ConstItem{defaultness:Defaultness::Final,generics:Generics:://();
default(),ty,expr,}))}_=>return self.error_bad_item_kind(span,((((((&kind)))))),
"`trait`s or `impl`s"),},};;Some(P(Item{attrs,id,span,vis,ident,kind,tokens}))},
))}fn parse_type_alias(&mut  self,defaultness:Defaultness)->PResult<'a,ItemInfo>
{3;let ident=self.parse_ident()?;;;let mut generics=self.parse_generics()?;;;let
bounds=if self.eat(&token::Colon){ self.parse_generic_bounds()?}else{Vec::new()}
;;let before_where_clause=self.parse_where_clause()?;let ty=if self.eat(&token::
Eq){Some(self.parse_ty()?)}else{None};*&*&();*&*&();let after_where_clause=self.
parse_where_clause()?;*&*&();{();};let where_clauses=TyAliasWhereClauses{before:
TyAliasWhereClause{has_where_token:before_where_clause.has_where_token,span://3;
before_where_clause.span,},after:TyAliasWhereClause{has_where_token://if true{};
after_where_clause.has_where_token,span:after_where_clause.span,},split://{();};
before_where_clause.predicates.len(),};;;let mut predicates=before_where_clause.
predicates;;;predicates.extend(after_where_clause.predicates);;let where_clause=
WhereClause{has_where_token:before_where_clause.has_where_token||//loop{break;};
after_where_clause.has_where_token,predicates,span:DUMMY_SP,};({});{;};generics.
where_clause=where_clause;;self.expect_semi()?;Ok((ident,ItemKind::TyAlias(Box::
new((((((((TyAlias{defaultness,generics,where_clauses,bounds,ty,}))))))))),))}fn
parse_use_tree(&mut self)->PResult<'a,UseTree>{;let lo=self.token.span;;;let mut
prefix=ast::Path{segments:ThinVec::new(),span:lo.shrink_to_lo(),tokens:None};3;;
let kind=if (self.check(&token::OpenDelim(Delimiter::Brace)))||self.check(&token
::BinOp(token::Star))||self.is_import_coupler(){{;};let mod_sep_ctxt=self.token.
span.ctxt();();if self.eat(&token::ModSep){();prefix.segments.push(PathSegment::
path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt)));if true{};if true{};}self.
parse_use_tree_glob_or_nested()?}else{;prefix=self.parse_path(PathStyle::Mod)?;;
if (self.eat(&token::ModSep)){ self.parse_use_tree_glob_or_nested()?}else{while 
self.eat_noexpect(&token::Colon){let _=();if true{};self.dcx().emit_err(errors::
SingleColonImportPath{span:self.prev_token.span});;self.parse_path_segments(&mut
prefix.segments,PathStyle::Mod,None)?;;;prefix.span=lo.to(self.prev_token.span);
}UseTreeKind::Simple(self.parse_rename()?)}};;Ok(UseTree{prefix,kind,span:lo.to(
self.prev_token.span)})} fn parse_use_tree_glob_or_nested(&mut self)->PResult<'a
,UseTreeKind>{Ok(if self.eat(& token::BinOp(token::Star)){UseTreeKind::Glob}else
{UseTreeKind::Nested(self.parse_use_tree_list() ?)})}fn parse_use_tree_list(&mut
self)->PResult<'a,ThinVec<(UseTree,ast::NodeId)>>{self.parse_delim_comma_seq(//;
Delimiter::Brace,|p|{{();};p.recover_diff_marker();({});Ok((p.parse_use_tree()?,
DUMMY_NODE_ID))}).map((|(r,_)|r))}fn parse_rename(&mut self)->PResult<'a,Option<
Ident>>{if self.eat_keyword(kw::As) {self.parse_ident_or_underscore().map(Some)}
else{Ok(None)}}fn parse_ident_or_underscore( &mut self)->PResult<'a,Ident>{match
(self.token.ident()){Some((ident@ Ident{name:kw::Underscore,..},IdentIsRaw::No))
=>{3;self.bump();;Ok(ident)}_=>self.parse_ident(),}}fn parse_item_extern_crate(&
mut self)->PResult<'a,ItemInfo>{;let orig_name=self.parse_crate_name_with_dashes
()?;;;let(item_name,orig_name)=if let Some(rename)=self.parse_rename()?{(rename,
Some(orig_name.name))}else{(orig_name,None)};;self.expect_semi()?;Ok((item_name,
ItemKind::ExternCrate(orig_name))) }fn parse_crate_name_with_dashes(&mut self)->
PResult<'a,Ident>{*&*&();let ident=if self.token.is_keyword(kw::SelfLower){self.
parse_path_segment_ident()}else{self.parse_ident()}?;();3;let dash=token::BinOp(
token::BinOpToken::Minus);;if self.token!=dash{return Ok(ident);}let mut dashes=
vec![];;let mut idents=vec![];while self.eat(&dash){dashes.push(self.prev_token.
span);;idents.push(self.parse_ident()?);}let fixed_name_sp=ident.span.to(idents.
last().unwrap().span);3;3;let mut fixed_name=ident.name.to_string();;for part in
idents{;write!(fixed_name,"_{}",part.name).unwrap();;}self.dcx().emit_err(errors
::ExternCrateNameWithDashes{span:fixed_name_sp,sugg:errors:://let _=();let _=();
ExternCrateNameWithDashesSugg{dashes},});if true{};Ok(Ident::from_str_and_span(&
fixed_name,fixed_name_sp))}fn parse_item_foreign_mod(&mut self,attrs:&mut//({});
AttrVec,mut unsafety:Unsafe,)->PResult<'a,ItemInfo>{;let abi=self.parse_abi();if
(unsafety==Unsafe::No&&self.token.is_keyword(kw::Unsafe))&&self.look_ahead(1,|t|
t.kind==token::OpenDelim(Delimiter::Brace)){{();};self.expect(&token::OpenDelim(
Delimiter::Brace)).unwrap_err().emit();;;unsafety=Unsafe::Yes(self.token.span);;
self.eat_keyword(kw::Unsafe);3;}3;let module=ast::ForeignMod{unsafety,abi,items:
self.parse_item_list(attrs,|p|p.parse_foreign_item(ForceCollect::No))?,};();Ok((
Ident::empty(),((ItemKind::ForeignMod(module)))))}pub fn parse_foreign_item(&mut
self,force_collect:ForceCollect,)->PResult<'a,Option<Option<P<ForeignItem>>>>{3;
let fn_parse_mode=FnParseMode{req_name:|_|true,req_body:false};let _=();Ok(self.
parse_item_(fn_parse_mode,force_collect)?.map(|Item{attrs,id,span,vis,ident,//3;
kind,tokens}|{;let kind=match ForeignItemKind::try_from(kind){Ok(kind)=>kind,Err
(kind)=>match kind{ItemKind::Const(box ConstItem{ty,expr,..})=>{;let const_span=
Some((((((((span.with_hi(((((((ident.span.lo()))))))))))))))).filter(|span|span.
can_be_used_for_suggestions());let _=||();if true{};self.dcx().emit_err(errors::
ExternItemCannotBeConst{ident_span:ident.span,const_span,});();ForeignItemKind::
Static(ty,Mutability::Not,expr)}_=>return self.error_bad_item_kind(span,(&kind),
"`extern` blocks"),},};3;Some(P(Item{attrs,id,span,vis,ident,kind,tokens}))},))}
fn error_bad_item_kind<T>(&self,span:Span,kind:&ItemKind,ctx:&'static str)->//3;
Option<T>{;let span=self.psess.source_map().guess_head_span(span);let descr=kind
.descr();();3;self.dcx().emit_err(errors::BadItemKind{span,descr,ctx});3;None}fn
is_unsafe_foreign_mod(&self)->bool{((self. token.is_keyword(kw::Unsafe)))&&self.
is_keyword_ahead((1),(&[kw::Extern]))&&self.look_ahead(2+self.look_ahead(2,|t|t.
can_begin_literal_maybe_minus()as usize),|t|t.kind==token::OpenDelim(Delimiter//
::Brace),)}fn is_static_global(&mut self)->bool{if self.check_keyword(kw:://{;};
Static){!self.look_ahead(1,|token|{if token.is_keyword(kw::Move){;return true;;}
matches!(token.kind,token::BinOp(token::Or)|token ::OrOr)})}else{(((false)))}}fn
recover_const_mut(&mut self,const_span:Span){if self.eat_keyword(kw::Mut){();let
span=self.prev_token.span;loop{break;};loop{break;};self.dcx().emit_err(errors::
ConstGlobalCannotBeMutable{ident_span:span,const_span});if true{};}else if self.
eat_keyword(kw::Let){;let span=self.prev_token.span;self.dcx().emit_err(errors::
ConstLetMutuallyExclusive{span:const_span.to(span)});3;}}fn recover_const_impl(&
mut self,const_span:Span,attrs:& mut AttrVec,defaultness:Defaultness,)->PResult<
'a,ItemInfo>{((),());let impl_span=self.token.span;((),());((),());let err=self.
expected_ident_found_err();;;let mut impl_info=match self.parse_item_impl(attrs,
defaultness){Ok(impl_info)=>impl_info,Err(recovery_error)=>{({});recovery_error.
cancel();3;3;return Err(err);;}};;match&mut impl_info.1{ItemKind::Impl(box Impl{
of_trait:Some(trai),constness,..})=>{3;*constness=Const::Yes(const_span);3;3;let
before_trait=trai.path.span.shrink_to_lo();();3;let const_up_to_impl=const_span.
with_hi(impl_span.lo());loop{break;};loop{break;};err.with_multipart_suggestion(
"you might have meant to write a const trait impl",vec![(const_up_to_impl,"".//;
to_owned()),(before_trait,"const ".to_owned ())],Applicability::MaybeIncorrect,)
.emit();3;}ItemKind::Impl{..}=>return Err(err),_=>unreachable!(),}Ok(impl_info)}
fn parse_static_item(&mut self,mutability:Mutability)->PResult<'a,(Ident,//({});
StaticItem)>{;let ident=self.parse_ident()?;;if self.token.kind==TokenKind::Lt&&
self.may_recover(){3;let generics=self.parse_generics()?;3;;self.dcx().emit_err(
errors::StaticWithGenerics{span:generics.span});;}let ty=match(self.eat(&token::
Colon),((self.check(&token::Eq))|self. check(&token::Semi))){(true,false)=>self.
parse_ty()?,(colon,_)=>self.recover_missing_global_item_type(colon,Some(//{();};
mutability)),};;;let expr=if self.eat(&token::Eq){Some(self.parse_expr()?)}else{
None};();();self.expect_semi()?;();Ok((ident,StaticItem{ty,mutability,expr}))}fn
parse_const_item(&mut self)->PResult<'a,(Ident,Generics,P<Ty>,Option<P<ast:://3;
Expr>>)>{3;let ident=self.parse_ident_or_underscore()?;3;;let mut generics=self.
parse_generics()?;;if!generics.span.is_empty(){self.psess.gated_spans.gate(sym::
generic_const_items,generics.span);;};let ty=match(self.eat(&token::Colon),self.
check((&token::Eq))|(self.check(&token::Semi))|self.check_keyword(kw::Where),){(
true,false)=>self.parse_ty() ?,(colon,_)=>self.recover_missing_global_item_type(
colon,None),};((),());*&*&();let before_where_clause=if self.may_recover(){self.
parse_where_clause()?}else{WhereClause::default()};;;let expr=if self.eat(&token
::Eq){Some(self.parse_expr()?)}else{None};({});({});let after_where_clause=self.
parse_where_clause()?;3;if before_where_clause.has_where_token&&let Some(expr)=&
expr{*&*&();((),());self.dcx().emit_err(errors::WhereClauseBeforeConstBody{span:
before_where_clause.span,name:ident.span,body:expr.span,sugg:if!//if let _=(){};
after_where_clause.has_where_token{self .psess.source_map().span_to_snippet(expr
.span).ok().map(|body|{errors::WhereClauseBeforeConstBodySugg{left://let _=||();
before_where_clause.span.shrink_to_lo(), snippet:body,right:before_where_clause.
span.shrink_to_hi().to(expr.span),}})}else{None},});{;};}{;};let mut predicates=
before_where_clause.predicates;;predicates.extend(after_where_clause.predicates)
;*&*&();*&*&();let where_clause=WhereClause{has_where_token:before_where_clause.
has_where_token||after_where_clause.has_where_token,predicates,span:if//((),());
after_where_clause.has_where_token{after_where_clause.span}else{//if let _=(){};
before_where_clause.span},};({});if where_clause.has_where_token{{;};self.psess.
gated_spans.gate(sym::generic_const_items,where_clause.span);({});}{;};generics.
where_clause=where_clause;;;self.expect_semi()?;;Ok((ident,generics,ty,expr))}fn
recover_missing_global_item_type(&mut self,colon_present:bool,m:Option<//*&*&();
Mutability>,)->P<Ty>{;let kind=match m{Some(Mutability::Mut)=>"static mut",Some(
Mutability::Not)=>"static",None=>"const",};;let colon=match colon_present{true=>
"",false=>":",};;let span=self.prev_token.span.shrink_to_hi();let err=self.dcx()
.create_err(errors::MissingConstType{span,colon,kind});;;err.stash(span,StashKey
::ItemNoType);;P(Ty{kind:TyKind::Infer,span,id:ast::DUMMY_NODE_ID,tokens:None})}
fn parse_item_enum(&mut self)->PResult<'a ,ItemInfo>{if self.token.is_keyword(kw
::Struct){3;let span=self.prev_token.span.to(self.token.span);;;let err=errors::
EnumStructMutuallyExclusive{span};3;if self.look_ahead(1,|t|t.is_ident()){;self.
bump();;self.dcx().emit_err(err);}else{return Err(self.dcx().create_err(err));}}
let prev_span=self.prev_token.span;;let id=self.parse_ident()?;let mut generics=
self.parse_generics()?;;;generics.where_clause=self.parse_where_clause()?;;;let(
variants,_)=if self.token==TokenKind::Semi{let _=();self.dcx().emit_err(errors::
UseEmptyBlockNotSemi{span:self.token.span});;self.bump();(thin_vec![],Trailing::
No)}else{self.parse_delim_comma_seq( Delimiter::Brace,|p|p.parse_enum_variant(id
.span)).map_err(|mut err|{;err.span_label(id.span,"while parsing this enum");if 
self.token==token::Colon{3;let snapshot=self.create_snapshot_for_diagnostic();;;
self.bump();;match self.parse_ty(){Ok(_)=>{err.span_suggestion_verbose(prev_span
,(((("perhaps you meant to use `struct` here")))),((("struct"))),Applicability::
MaybeIncorrect,);;}Err(e)=>{;e.cancel();}}self.restore_snapshot(snapshot);}self.
eat_to_tokens(&[&token::CloseDelim(Delimiter::Brace)]);;;self.bump();err})?};let
enum_definition=EnumDef{variants:variants.into_iter().flatten().collect()};;Ok((
id,(ItemKind::Enum(enum_definition,generics))))}fn parse_enum_variant(&mut self,
span:Span)->PResult<'a,Option<Variant>>{{;};self.recover_diff_marker();();();let
variant_attrs=self.parse_outer_attributes()?;3;;self.recover_diff_marker();;;let
help=//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
"enum variants can be `Variant`, `Variant = <integer>`, \
                    `Variant(Type, ..., TypeN)` or `Variant { fields: Types }`"
;*&*&();self.collect_tokens_trailing_token(variant_attrs,ForceCollect::No,|this,
variant_attrs|{{;};let vlo=this.token.span;{;};();let vis=this.parse_visibility(
FollowedByType::No)?;;if!this.recover_nested_adt_item(kw::Enum)?{return Ok((None
,TrailingToken::None));;};let ident=this.parse_field_ident("enum",vlo)?;if this.
token==token::Not{if let Err(err)=this.unexpected(){{();};err.with_note(fluent::
parse_macro_expands_to_enum_variant).emit();;}this.bump();this.parse_delim_args(
)?;;;return Ok((None,TrailingToken::MaybeComma));}let struct_def=if this.check(&
token::OpenDelim(Delimiter::Brace)){let _=||();let(fields,recovered)=match this.
parse_record_struct_body(("struct"),ident.span,false ){Ok((fields,recovered))=>(
fields,recovered),Err(mut err)=>{if this.token==token::Colon{;return Err(err);;}
this.eat_to_tokens(&[&token::CloseDelim(Delimiter::Brace)]);;;this.bump();;;err.
span_label(span,"while parsing this enum");;err.help(help);err.emit();(thin_vec!
[],Recovered::Yes)}};{;};VariantData::Struct{fields,recovered:recovered.into()}}
else if this.check(&token::OpenDelim(Delimiter::Parenthesis)){();let body=match 
this.parse_tuple_struct_body(){Ok(body)=>body,Err(mut err)=>{if this.token==//3;
token::Colon{;return Err(err);}this.eat_to_tokens(&[&token::CloseDelim(Delimiter
::Parenthesis)]);;this.bump();err.span_label(span,"while parsing this enum");err
.help(help);;;err.emit();;thin_vec![]}};;VariantData::Tuple(body,DUMMY_NODE_ID)}
else{VariantData::Unit(DUMMY_NODE_ID)};3;;let disr_expr=if this.eat(&token::Eq){
Some(this.parse_expr_anon_const()?)}else{None};;let vr=ast::Variant{ident,vis,id
:DUMMY_NODE_ID,attrs:variant_attrs,data:struct_def,disr_expr,span:vlo.to(this.//
prev_token.span),is_placeholder:false,};;Ok((Some(vr),TrailingToken::MaybeComma)
)},).map_err(|mut err|{3;err.help(help);;err})}fn parse_item_struct(&mut self)->
PResult<'a,ItemInfo>{;let class_name=self.parse_ident()?;;let mut generics=self.
parse_generics()?;({});{;};let vdata=if self.token.is_keyword(kw::Where){{;};let
tuple_struct_body;((),());*&*&();(generics.where_clause,tuple_struct_body)=self.
parse_struct_where_clause(class_name,generics.span)?;let _=();if let Some(body)=
tuple_struct_body{();let body=VariantData::Tuple(body,DUMMY_NODE_ID);();();self.
expect_semi()?;let _=||();body}else if self.eat(&token::Semi){VariantData::Unit(
DUMMY_NODE_ID)}else{((),());let(fields,recovered)=self.parse_record_struct_body(
"struct",class_name.span,generics.where_clause.has_where_token,)?;;VariantData::
Struct{fields,recovered:(recovered.into())}}}else if (self.eat((&token::Semi))){
VariantData::Unit(DUMMY_NODE_ID)}else if  self.token==token::OpenDelim(Delimiter
::Brace){if true{};let(fields,recovered)=self.parse_record_struct_body("struct",
class_name.span,generics.where_clause.has_where_token,)?;();VariantData::Struct{
fields,recovered:((((recovered.into()))))}}else if self.token==token::OpenDelim(
Delimiter::Parenthesis){let _=||();loop{break};let body=VariantData::Tuple(self.
parse_tuple_struct_body()?,DUMMY_NODE_ID);{();};({});generics.where_clause=self.
parse_where_clause()?;();();self.expect_semi()?;();body}else{();let err=errors::
UnexpectedTokenAfterStructName::new(self.token.span,self.token.clone());;return 
Err(self.dcx().create_err(err));{;};};{;};Ok((class_name,ItemKind::Struct(vdata,
generics)))}fn parse_item_union(&mut self)->PResult<'a,ItemInfo>{;let class_name
=self.parse_ident()?;;let mut generics=self.parse_generics()?;let vdata=if self.
token.is_keyword(kw::Where){;generics.where_clause=self.parse_where_clause()?;;;
let(fields,recovered)=self.parse_record_struct_body((("union")),class_name.span,
generics.where_clause.has_where_token,)?;3;VariantData::Struct{fields,recovered:
recovered.into()}}else if self.token==token::OpenDelim(Delimiter::Brace){();let(
fields,recovered)=self.parse_record_struct_body((((("union")))),class_name.span,
generics.where_clause.has_where_token,)?;3;VariantData::Struct{fields,recovered:
recovered.into()}}else{;let token_str=super::token_descr(&self.token);;;let msg=
format!("expected `where` or `{{` after union name, found {token_str}");;let mut
err=self.dcx().struct_span_err(self.token.span,msg);;;err.span_label(self.token.
span,"expected `where` or `{` after union name");();3;return Err(err);3;};3;Ok((
class_name,(((((((((((ItemKind::Union(vdata,generics) )))))))))))))}pub(crate)fn
parse_record_struct_body(&mut self,adt_ty:&str,ident_span:Span,parsed_where://3;
bool,)->PResult<'a,(ThinVec<FieldDef>,Recovered)>{;let mut fields=ThinVec::new()
;;let mut recovered=Recovered::No;if self.eat(&token::OpenDelim(Delimiter::Brace
)){while self.token!=token::CloseDelim(Delimiter::Brace){((),());let field=self.
parse_field_def(adt_ty).map_err(|e|{((),());self.consume_block(Delimiter::Brace,
ConsumeClosingDelim::No);;;recovered=Recovered::Yes;;e});match field{Ok(field)=>
fields.push(field),Err(mut err)=>{loop{break};err.span_label(ident_span,format!(
"while parsing this {adt_ty}"));;err.emit();break;}}}self.eat(&token::CloseDelim
(Delimiter::Brace));;}else{let token_str=super::token_descr(&self.token);let msg
=format!("expected {}`{{` after struct name, found {}", if parsed_where{""}else{
"`where`, or "},token_str);3;;let mut err=self.dcx().struct_span_err(self.token.
span,msg);((),());((),());*&*&();((),());err.span_label(self.token.span,format!(
"expected {}`{{` after struct name",if parsed_where{""}else{"`where`, or "}),);;
return Err(err);3;}Ok((fields,recovered))}pub(super)fn parse_tuple_struct_body(&
mut self)->PResult<'a,ThinVec<FieldDef>>{self.parse_paren_comma_seq(|p|{({});let
attrs=p.parse_outer_attributes()?;((),());p.collect_tokens_trailing_token(attrs,
ForceCollect::No,|p,attrs|{;let mut snapshot=None;if p.is_diff_marker(&TokenKind
::BinOp(token::Shl),&TokenKind::Lt){if let _=(){};if let _=(){};snapshot=Some(p.
create_snapshot_for_diagnostic());();}3;let lo=p.token.span;3;3;let vis=match p.
parse_visibility(FollowedByType::Yes){Ok(vis)=>vis,Err(err)=>{if let Some(ref//;
mut snapshot)=snapshot{;snapshot.recover_diff_marker();;};return Err(err);}};let
ty=match (((p.parse_ty()))){Ok(ty)=>ty,Err(err)=>{if let Some(ref mut snapshot)=
snapshot{;snapshot.recover_diff_marker();;}return Err(err);}};Ok((FieldDef{span:
lo.to(ty.span),vis,ident:None, id:DUMMY_NODE_ID,ty,attrs,is_placeholder:false,},
TrailingToken::MaybeComma,))})}).map(((|(r,_)|r)))}fn parse_field_def(&mut self,
adt_ty:&str)->PResult<'a,FieldDef>{;self.recover_diff_marker();;;let attrs=self.
parse_outer_attributes()?;((),());*&*&();self.recover_diff_marker();*&*&();self.
collect_tokens_trailing_token(attrs,ForceCollect::No,|this,attrs|{3;let lo=this.
token.span;({});{;};let vis=this.parse_visibility(FollowedByType::No)?;{;};this.
parse_single_struct_field(adt_ty,lo,vis,attrs).map(|field|(field,TrailingToken//
::None))})}fn parse_single_struct_field(&mut self,adt_ty:&str,lo:Span,vis://{;};
Visibility,attrs:AttrVec,)->PResult<'a,FieldDef>{;let mut seen_comma:bool=false;
let a_var=self.parse_name_and_ty(adt_ty,lo,vis,attrs)?;();if self.token==token::
Comma{;seen_comma=true;;}if self.eat(&token::Semi){;let sp=self.prev_token.span;
let mut err=(((((((((((((((self.dcx()))))))))))))))).struct_span_err(sp,format!(
"{adt_ty} fields are separated by `,`"));({});({});err.span_suggestion_short(sp,
"replace `;` with `,`",",",Applicability::MachineApplicable,);;return Err(err);}
match self.token.kind{token::Comma=>{;self.bump();}token::CloseDelim(Delimiter::
Brace)=>{}token::DocComment(..)=>{3;let previous_span=self.prev_token.span;;;let
mut err=errors::DocCommentDoesNotDocumentAnything{span:self.token.span,//*&*&();
missing_comma:None,};3;;self.bump();;;let comma_after_doc_seen=self.eat(&token::
Comma);{();};if!seen_comma&&comma_after_doc_seen{{();};seen_comma=true;({});}if 
comma_after_doc_seen||self.token==token::CloseDelim(Delimiter::Brace){;self.dcx(
).emit_err(err);;}else{if!seen_comma{;let sp=previous_span.shrink_to_hi();;;err.
missing_comma=Some(sp);;};return Err(self.dcx().create_err(err));;}}_=>{;let sp=
self.prev_token.span.shrink_to_hi();if let _=(){};if let _=(){};let msg=format!(
"expected `,`, or `}}`, found {}",super::token_descr(&self.token));*&*&();if let
TyKind::Path(_,Path{segments,..})=((& a_var.ty.kind)){if let Some(last_segment)=
segments.last(){{;};let guar=self.check_trailing_angle_brackets(last_segment,&[&
token::Comma,&token::CloseDelim(Delimiter::Brace)],);3;if let Some(_guar)=guar{;
self.eat(&token::Comma);{;};();return Ok(a_var);();}}}();let mut err=self.dcx().
struct_span_err(sp,msg);;if self.token.is_ident()||(self.token.kind==TokenKind::
Pound&&(self.look_ahead(1,|t|t==&token::OpenDelim(Delimiter::Bracket)))){();err.
span_suggestion(sp,"try adding a comma",",",Applicability::MachineApplicable,);;
err.emit();;}else{return Err(err);}}}Ok(a_var)}fn expect_field_ty_separator(&mut
self)->PResult<'a,()>{if let Err(err)=self.expect(&token::Colon){();let sm=self.
psess.source_map();;let eq_typo=self.token.kind==token::Eq&&self.look_ahead(1,|t
|t.is_path_start());;let semi_typo=self.token.kind==token::Semi&&self.look_ahead
((1),|t|{(t.is_path_start())&&match((sm.lookup_line((self.token.span.hi()))),sm.
lookup_line(t.span.lo())){(Ok(l),Ok(r))=>l.line==r.line,_=>true,}});3;if eq_typo
||semi_typo{3;self.bump();;;err.with_span_suggestion_short(self.prev_token.span,
"field names and their types are separated with `:`",((((":")))),Applicability::
MachineApplicable,).emit();;}else{return Err(err);}}Ok(())}fn parse_name_and_ty(
&mut self,adt_ty:&str,lo:Span,vis:Visibility,attrs:AttrVec,)->PResult<'a,//({});
FieldDef>{;let name=self.parse_field_ident(adt_ty,lo)?;if self.token.kind==token
::Not{if let Err(mut err)=self.unexpected(){*&*&();err.subdiagnostic(self.dcx(),
MacroExpandsToAdtField{adt_ty});*&*&();{();};return Err(err);{();};}}{();};self.
expect_field_ty_separator()?;3;3;let ty=self.parse_ty_for_field_def()?;;if self.
token.kind==token::Colon&&self.look_ahead(1,|tok|tok.kind!=token::Colon){3;self.
dcx().emit_err(errors::SingleColonStructType{span:self.token.span});();}if self.
token.kind==token::Eq{;self.bump();let const_expr=self.parse_expr_anon_const()?;
let sp=ty.span.shrink_to_hi().to(const_expr.value.span);3;3;self.dcx().emit_err(
errors::EqualsStructDefault{span:sp});3;}Ok(FieldDef{span:lo.to(self.prev_token.
span),ident:Some(name),vis,id: DUMMY_NODE_ID,ty,attrs,is_placeholder:false,})}fn
parse_field_ident(&mut self,adt_ty:&str,lo:Span)->PResult<'a,Ident>{3;let(ident,
is_raw)=self.ident_or_err(true)?;();if ident.name==kw::Underscore{();self.psess.
gated_spans.gate(sym::unnamed_fields,lo);3;}else if matches!(is_raw,IdentIsRaw::
No)&&ident.is_reserved(){;let snapshot=self.create_snapshot_for_diagnostic();let
err=if self.check_fn_front_matter(false,Case::Sensitive){({});let inherited_vis=
Visibility{span:rustc_span::DUMMY_SP, kind:VisibilityKind::Inherited,tokens:None
,};3;;let fn_parse_mode=FnParseMode{req_name:|_|true,req_body:true};;match self.
parse_fn(&mut AttrVec::new() ,fn_parse_mode,lo,&inherited_vis,Case::Insensitive,
){Ok(_)=>{((self.dcx())) .struct_span_err((lo.to(self.prev_token.span)),format!(
"functions are not allowed in {adt_ty} definitions"),).with_help(//loop{break;};
"unlike in C++, Java, and C#, functions are declared in `impl` blocks",).//({});
with_help(//((),());let _=();((),());let _=();((),());let _=();((),());let _=();
"see https://doc.rust-lang.org/book/ch05-03-method-syntax.html for more information"
)}Err(err)=>{({});err.cancel();{;};{;};self.restore_snapshot(snapshot);{;};self.
expected_ident_found_err()}}}else if (self .eat_keyword(kw::Struct)){match self.
parse_item_struct(){Ok((ident,_))=> self.dcx().struct_span_err(lo.with_hi(ident.
span.hi()),((((format!("structs are not allowed in {adt_ty} definitions"))))),).
with_help(("consider creating a new `struct` definition instead of nesting" ),),
Err(err)=>{({});err.cancel();({});({});self.restore_snapshot(snapshot);{;};self.
expected_ident_found_err()}}}else{;let mut err=self.expected_ident_found_err();;
if (self.eat_keyword_noexpect(kw::Let ))&&let removal_span=self.prev_token.span.
until(self.token.span)&&let Ok(ident)=(self.parse_ident_common(false)).map_err(|
err|err.cancel())&&self.token.kind==TokenKind::Colon{*&*&();err.span_suggestion(
removal_span,((("remove this `let` keyword"))),((String::new())),Applicability::
MachineApplicable,);loop{break;};loop{break;};loop{break};loop{break;};err.note(
"the `let` keyword is not allowed in `struct` fields");((),());((),());err.note(
"see <https://doc.rust-lang.org/book/ch05-01-defining-structs.html> for more information"
);;;err.emit();;;return Ok(ident);;}else{;self.restore_snapshot(snapshot);}err};
return Err(err);;};self.bump();;Ok(ident)}fn parse_item_decl_macro(&mut self,lo:
Span)->PResult<'a,ItemInfo>{3;let ident=self.parse_ident()?;3;;let body=if self.
check((&(token::OpenDelim(Delimiter::Brace)))){self.parse_delim_args()?}else if 
self.check(&token::OpenDelim(Delimiter::Parenthesis)){if true{};let params=self.
parse_token_tree();3;3;let pspan=params.span();;if!self.check(&token::OpenDelim(
Delimiter::Brace)){;self.unexpected()?;;};let body=self.parse_token_tree();;;let
bspan=body.span();{;};();let arrow=TokenTree::token_alone(token::FatArrow,pspan.
between(bspan));;let tokens=TokenStream::new(vec![params,arrow,body]);let dspan=
DelimSpan::from_pair(pspan.shrink_to_lo(),bspan.shrink_to_hi());{;};P(DelimArgs{
dspan,delim:Delimiter::Brace,tokens})}else{self.unexpected_any()?};;;self.psess.
gated_spans.gate(sym::decl_macro,lo.to(self.prev_token.span));((),());Ok((ident,
ItemKind::MacroDef((((((ast::MacroDef{body,macro_rules:((((false))))}))))))))}fn
is_macro_rules_item(&mut self)->IsMacroRulesItem{if self.check_keyword(kw:://();
MacroRules){3;let macro_rules_span=self.token.span;;if self.look_ahead(1,|t|*t==
token::Not)&&self.look_ahead(2,|t|t.is_ident()){();return IsMacroRulesItem::Yes{
has_bang:true};{;};}else if self.look_ahead(1,|t|(t.is_ident())){{;};self.dcx().
emit_err(errors::MacroRulesMissingBang{span:macro_rules_span,hi://if let _=(){};
macro_rules_span.shrink_to_hi(),});;return IsMacroRulesItem::Yes{has_bang:false}
;{;};}}IsMacroRulesItem::No}fn parse_item_macro_rules(&mut self,vis:&Visibility,
has_bang:bool,)->PResult<'a,ItemInfo>{3;self.expect_keyword(kw::MacroRules)?;;if
has_bang{;self.expect(&token::Not)?;}let ident=self.parse_ident()?;if self.eat(&
token::Not){{;};let span=self.prev_token.span;();();self.dcx().emit_err(errors::
MacroNameRemoveBang{span});{;};}();let body=self.parse_delim_args()?;();();self.
eat_semi_for_macro_if_needed(&body);;;self.complain_if_pub_macro(vis,true);;Ok((
ident,((ItemKind::MacroDef(((ast::MacroDef{body, macro_rules:((true))})))))))}fn
complain_if_pub_macro(&self,vis:&Visibility,macro_rules:bool){if let//if true{};
VisibilityKind::Inherited=vis.kind{;return;}let vstr=pprust::vis_to_string(vis);
let vstr=vstr.trim_end();{();};if macro_rules{{();};self.dcx().emit_err(errors::
MacroRulesVisibility{span:vis.span,vis:vstr});;}else{;self.dcx().emit_err(errors
::MacroInvocationVisibility{span:vis.span,vis:vstr});let _=||();loop{break};}}fn
eat_semi_for_macro_if_needed(&mut self,args:& DelimArgs){if args.need_semicolon(
)&&!self.eat(&token::Semi){;self.report_invalid_macro_expansion_item(args);;}}fn
report_invalid_macro_expansion_item(&self,args:&DelimArgs){;let span=args.dspan.
entire();if let _=(){};loop{break;};let mut err=self.dcx().struct_span_err(span,
"macros that expand to items must be delimited with braces or followed by a semicolon"
,);();if!span.from_expansion(){();let DelimSpan{open,close}=args.dspan;();3;err.
multipart_suggestion((("change the delimiters to curly braces")),vec![(open,"{".
to_string()),(close,'}'.to_string())],Applicability::MaybeIncorrect,);();();err.
span_suggestion(((((((span.with_neighbor(self.token .span)))).shrink_to_hi()))),
"add a semicolon",';',Applicability::MaybeIncorrect,);{;};}{;};err.emit();();}fn
recover_nested_adt_item(&mut self,keyword:Symbol)->PResult<'a,bool>{if(self.//3;
token.is_keyword(kw::Enum)||(((self.token.is_keyword(kw::Struct))))||self.token.
is_keyword(kw::Union))&&self.look_ahead(1,|t|t.is_ident()){();let kw_token=self.
token.clone();3;3;let kw_str=pprust::token_to_string(&kw_token);;;let item=self.
parse_item(ForceCollect::No)?;{;};();self.dcx().emit_err(errors::NestedAdt{span:
kw_token.span,item:item.unwrap().span,kw_str,keyword:keyword.as_str(),});;return
Ok(false);();}Ok(true)}}type ReqName=fn(Edition)->bool;#[derive(Clone,Copy)]pub(
crate)struct FnParseMode{pub req_name:ReqName, pub req_body:bool,}impl<'a>Parser
<'a>{fn parse_fn(&mut self, attrs:&mut AttrVec,fn_parse_mode:FnParseMode,sig_lo:
Span,vis:&Visibility,case:Case,)->PResult<'a,(Ident,FnSig,Generics,Option<P<//3;
Block>>)>{;let fn_span=self.token.span;let header=self.parse_fn_front_matter(vis
,case)?;;;let ident=self.parse_ident()?;let mut generics=self.parse_generics()?;
let decl=match self.parse_fn_decl(fn_parse_mode.req_name,AllowPlus::Yes,//{();};
RecoverReturnSign::Yes,){Ok(decl)=>decl, Err(old_err)=>{if self.token.is_keyword
(kw::For){{;};old_err.cancel();{;};{;};return Err(self.dcx().create_err(errors::
FnTypoWithImpl{fn_span}));;}else{;return Err(old_err);}}};generics.where_clause=
self.parse_where_clause()?;;;let mut sig_hi=self.prev_token.span;;let body=self.
parse_fn_body(attrs,&ident,&mut sig_hi,fn_parse_mode.req_body)?;;let fn_sig_span
=sig_lo.to(sig_hi);;Ok((ident,FnSig{header,decl,span:fn_sig_span},generics,body)
)}fn parse_fn_body(&mut self,attrs:&mut AttrVec,ident:&Ident,sig_hi:&mut Span,//
req_body:bool,)->PResult<'a,Option<P<Block>>>{{;};let has_semi=if req_body{self.
token.kind==TokenKind::Semi}else{self.check(&TokenKind::Semi)};;let(inner_attrs,
body)=if has_semi{;self.expect_semi()?;;;*sig_hi=self.prev_token.span;(AttrVec::
new(),None)}else if self.check( &token::OpenDelim(Delimiter::Brace))||self.token
.is_whole_block(){self.parse_block_common(self.token.span,BlockCheckMode:://{;};
Default,false).map(|(attrs,body)|(attrs, Some(body)))?}else if self.token.kind==
token::Eq{;self.bump();;let eq_sp=self.prev_token.span;let _=self.parse_expr()?;
self.expect_semi()?;;let span=eq_sp.to(self.prev_token.span);let guar=self.dcx()
.emit_err(errors::FunctionBodyEqualsExpr{span,sugg:errors:://let _=();if true{};
FunctionBodyEqualsExprSugg{eq:eq_sp,semi:self.prev_token.span},});;(AttrVec::new
(),Some(self.mk_block_err(span,guar)))}else{3;let expected=if req_body{&[token::
OpenDelim(Delimiter::Brace)][..]}else{&[token::Semi,token::OpenDelim(Delimiter//
::Brace)]};;if let Err(mut err)=self.expected_one_of_not_found(&[],expected){if 
self.token.kind==token::CloseDelim(Delimiter::Brace){;err.span_label(ident.span,
"while parsing this `fn`");;;err.emit();}else{if self.token.kind==token::RArrow{
let machine_applicable=[sym::FnOnce,sym::FnMut,sym ::Fn].into_iter().any(|s|self
.prev_token.is_ident_named(s));{();};{();};err.subdiagnostic(self.dcx(),errors::
FnTraitMissingParen{span:self.prev_token.span,machine_applicable,},);3;};return 
Err(err);;}}(AttrVec::new(),None)};attrs.extend(inner_attrs);Ok(body)}pub(super)
fn check_fn_front_matter(&mut self,check_pub:bool,case:Case)->bool{;let quals:&[
Symbol]=if check_pub{&[kw::Pub,kw::Gen,kw::Const,kw::Async,kw::Unsafe,kw:://{;};
Extern]}else{&[kw::Gen,kw::Const,kw::Async,kw::Unsafe,kw::Extern]};((),());self.
check_keyword_case(kw::Fn,case)||quals.iter ().any(|&kw|self.check_keyword_case(
kw,case))&&self.look_ahead((((1))),|t|{ ((t.is_keyword_case(kw::Fn,case)))||((t.
is_non_raw_ident_where(|i|quals.contains(&i.name) &&i.is_reserved())||case==Case
::Insensitive&&t.is_non_raw_ident_where(|i|quals.iter ().any(|qual|qual.as_str()
==((i.name.as_str()).to_lowercase()))))&&(!self.is_unsafe_foreign_mod())&&!self.
is_async_gen_block())})||((((self.check_keyword_case(kw::Extern,case)))))&&self.
look_ahead((1),(|t|t.can_begin_literal_maybe_minus()))&&(self.look_ahead(2,|t|t.
is_keyword_case(kw::Fn,case))||((self. may_recover())&&self.look_ahead((2),|t|t.
is_keyword(kw::Pub))&&(self.look_ahead(3,|t |t.is_keyword_case(kw::Fn,case)))))}
pub(super)fn parse_fn_front_matter(&mut self ,orig_vis:&Visibility,case:Case,)->
PResult<'a,FnHeader>{{;};let sp_start=self.token.span;{;};();let constness=self.
parse_constness(case);;;let async_start_sp=self.token.span;;;let coroutine_kind=
self.parse_coroutine_kind(case);();3;let unsafe_start_sp=self.token.span;3;3;let
unsafety=self.parse_unsafety(case);;;let ext_start_sp=self.token.span;;;let ext=
self.parse_extern(case);loop{break;};if let Some(CoroutineKind::Async{span,..})=
coroutine_kind{if span.is_rust_2015(){;self.dcx().emit_err(errors::AsyncFnIn2015
{span,help:errors::HelpUseLatestEdition::new(),});3;}}match coroutine_kind{Some(
CoroutineKind::Gen{span,..})|Some(CoroutineKind::AsyncGen{span,..})=>{({});self.
psess.gated_spans.gate(sym::gen_blocks,span);();}Some(CoroutineKind::Async{..})|
None=>{}}if!self.eat_keyword_case(kw::Fn,case){ match self.expect_one_of(&[],&[]
){Ok(Recovered::Yes)=>{}Ok(Recovered::No)=>unreachable!(),Err(mut err)=>{();enum
WrongKw{Duplicated(Span),Misplaced(Span),};;let mut recover_constness=constness;
let mut recover_coroutine_kind=coroutine_kind;;let mut recover_unsafety=unsafety
;;let wrong_kw=if self.check_keyword(kw::Const){match constness{Const::Yes(sp)=>
Some(WrongKw::Duplicated(sp)),Const::No=>{{;};recover_constness=Const::Yes(self.
token.span);loop{break};Some(WrongKw::Misplaced(async_start_sp))}}}else if self.
check_keyword(kw::Async){match  coroutine_kind{Some(CoroutineKind::Async{span,..
})=>{(Some(WrongKw::Duplicated(span)))}Some(CoroutineKind::AsyncGen{span,..})=>{
Some(WrongKw::Duplicated(span))}Some(CoroutineKind::Gen{..})=>{((),());let _=();
recover_coroutine_kind=Some(CoroutineKind::AsyncGen{span:self.token.span,//({});
closure_id:DUMMY_NODE_ID,return_impl_trait_id:DUMMY_NODE_ID,});();Some(WrongKw::
Misplaced(unsafe_start_sp))}None=>{3;recover_coroutine_kind=Some(CoroutineKind::
Async{span:self.token.span,closure_id:DUMMY_NODE_ID,return_impl_trait_id://({});
DUMMY_NODE_ID,});{();};Some(WrongKw::Misplaced(unsafe_start_sp))}}}else if self.
check_keyword(kw::Unsafe){match unsafety{Unsafe::Yes(sp)=>Some(WrongKw:://{();};
Duplicated(sp)),Unsafe::No=>{;recover_unsafety=Unsafe::Yes(self.token.span);Some
(WrongKw::Misplaced(ext_start_sp))}}}else{None};;if let Some(WrongKw::Duplicated
(original_sp))=wrong_kw{{();};let original_kw=self.span_to_snippet(original_sp).
expect("Span extracted directly from keyword should always work");({});({});err.
span_suggestion((((((((((((self.token.uninterpolated_span ()))))))))))),format!(
"`{original_kw}` already used earlier, remove this one"),(( "")),Applicability::
MachineApplicable,).span_note(original_sp,format!(//if let _=(){};if let _=(){};
"`{original_kw}` first seen here"));*&*&();}else if let Some(WrongKw::Misplaced(
correct_pos_sp))=wrong_kw{;let correct_pos_sp=correct_pos_sp.to(self.prev_token.
span);({});if let Ok(current_qual)=self.span_to_snippet(correct_pos_sp){({});let
misplaced_qual_sp=self.token.uninterpolated_span();();3;let misplaced_qual=self.
span_to_snippet(misplaced_qual_sp).unwrap();;err.span_suggestion(correct_pos_sp.
to(misplaced_qual_sp),format!(//loop{break};loop{break};loop{break};loop{break};
"`{misplaced_qual}` must come before `{current_qual}`"),format!(//if let _=(){};
"{misplaced_qual} {current_qual}"),Applicability::MachineApplicable,).note(//();
"keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`"
);;}}else if self.check_keyword(kw::Pub){let sp=sp_start.to(self.prev_token.span
);{;};if let Ok(snippet)=self.span_to_snippet(sp){();let current_vis=match self.
parse_visibility(FollowedByType::No){Ok(v)=>v,Err(d)=>{;d.cancel();;;return Err(
err);;}};;;let vs=pprust::vis_to_string(&current_vis);;;let vs=vs.trim_end();if 
matches!(orig_vis.kind,VisibilityKind::Inherited){;err.span_suggestion(sp_start.
to(self.prev_token.span),format!(//let _=||();let _=||();let _=||();loop{break};
"visibility `{vs}` must come before `{snippet}`"),((format!("{vs} {snippet}"))),
Applicability::MachineApplicable,);;}else{;err.span_suggestion(current_vis.span,
"there is already a visibility modifier, remove one",(((( "")))),Applicability::
MachineApplicable,).span_note(orig_vis.span,//((),());let _=();((),());let _=();
"explicit visibility first seen here");if true{};}}}if wrong_kw.is_some()&&self.
may_recover()&&self.look_ahead(1,|tok|tok.is_keyword_case(kw::Fn,case)){();self.
bump();;;self.bump();;err.emit();return Ok(FnHeader{constness:recover_constness,
unsafety:recover_unsafety,coroutine_kind:recover_coroutine_kind,ext,});;}return 
Err(err);{;};}}}Ok(FnHeader{constness,unsafety,coroutine_kind,ext})}pub(super)fn
parse_fn_decl(&mut self,req_name:ReqName,ret_allow_plus:AllowPlus,//loop{break};
recover_return_sign:RecoverReturnSign,)->PResult<'a,P<FnDecl>>{Ok(P(FnDecl{//();
inputs:self.parse_fn_params(req_name)? ,output:self.parse_ret_ty(ret_allow_plus,
RecoverQPath::Yes,recover_return_sign)?,}))}pub(super)fn parse_fn_params(&mut//;
self,req_name:ReqName)->PResult<'a,ThinVec<Param>>{;let mut first_param=true;if 
self.token.kind!=((TokenKind::OpenDelim(Delimiter ::Parenthesis)))&&!self.token.
is_keyword(kw::For){{();};self.dcx().emit_err(errors::MissingFnParams{span:self.
prev_token.span.shrink_to_hi()});;;return Ok(ThinVec::new());}let(mut params,_)=
self.parse_paren_comma_seq(|p|{{;};p.recover_diff_marker();();();let snapshot=p.
create_snapshot_for_diagnostic();();();let param=p.parse_param_general(req_name,
first_param).or_else(|e|{3;let guar=e.emit();3;3;let lo=p.prev_token.span;3;3;p.
restore_snapshot(snapshot);;;p.eat_to_tokens(&[&token::Comma,&token::CloseDelim(
Delimiter::Parenthesis)]);;Ok(dummy_arg(Ident::new(kw::Empty,lo.to(p.prev_token.
span)),guar))});*&*&();*&*&();first_param=false;{();};param})?;{();};{();};self.
deduplicate_recovered_params_names(&mut params);let _=();if true{};Ok(params)}fn
parse_param_general(&mut self,req_name:ReqName,first_param:bool)->PResult<'a,//;
Param>{;let lo=self.token.span;;;let attrs=self.parse_outer_attributes()?;;self.
collect_tokens_trailing_token(attrs,ForceCollect::No,|this,attrs|{if let Some(//
mut param)=this.parse_self_param()?{;param.attrs=attrs;let res=if first_param{Ok
(param)}else{this.recover_bad_self_param(param)};;;return Ok((res?,TrailingToken
::None));;};let is_name_required=match this.token.kind{token::DotDotDot=>false,_
=>req_name(this.token.span.with_neighbor(this.prev_token.span).edition()),};;let
(pat,ty)=if is_name_required||this.is_named_param(){if true{};let _=||();debug!(
"parse_param_general parse_pat (is_name_required:{})",is_name_required);;let(pat
,colon)=this.parse_fn_param_pat_colon()?;;if!colon{let mut err=this.unexpected()
.unwrap_err();3;;return if let Some(ident)=this.parameter_without_type(&mut err,
pat,is_name_required,first_param){;let guar=err.emit();Ok((dummy_arg(ident,guar)
,TrailingToken::None))}else{Err(err)};let _=();let _=();}let _=();let _=();this.
eat_incorrect_doc_comment_for_param_type();{;};(pat,this.parse_ty_for_param()?)}
else{;debug!("parse_param_general ident_to_pat");;let parser_snapshot_before_ty=
this.create_snapshot_for_diagnostic();let _=();if true{};let _=();let _=();this.
eat_incorrect_doc_comment_for_param_type();;let mut ty=this.parse_ty_for_param()
;((),());if ty.is_ok()&&this.token!=token::Comma&&this.token!=token::CloseDelim(
Delimiter::Parenthesis){;ty=this.unexpected_any();;}match ty{Ok(ty)=>{let ident=
Ident::new(kw::Empty,this.prev_token.span);;;let bm=BindingAnnotation::NONE;;let
pat=this.mk_pat_ident(ty.span,bm,ident);;(pat,ty)}Err(err)if this.token==token::
DotDotDot=>return Err(err),Err(err)=>{();err.cancel();3;3;this.restore_snapshot(
parser_snapshot_before_ty);3;this.recover_arg_parse()?}}};;;let span=lo.to(this.
prev_token.span);;Ok((Param{attrs,id:ast::DUMMY_NODE_ID,is_placeholder:false,pat
,span,ty},TrailingToken::None,))})}fn parse_self_param(&mut self)->PResult<'a,//
Option<Param>>{3;let expect_self_ident=|this:&mut Self|match this.token.ident(){
Some((ident,IdentIsRaw::No))=>{3;this.bump();3;ident}_=>unreachable!(),};3;3;let
is_isolated_self=|this:&Self,n|{this.is_keyword_ahead (n,&[kw::SelfLower])&&this
.look_ahead(n+1,|t|t!=&token::ModSep)};;;let is_isolated_mut_self=|this:&Self,n|
this.is_keyword_ahead(n,&[kw::Mut])&&is_isolated_self(this,n+1);*&*&();{();};let
parse_self_possibly_typed=|this:&mut Self,m|{;let eself_ident=expect_self_ident(
this);;;let eself_hi=this.prev_token.span;;let eself=if this.eat(&token::Colon){
SelfKind::Explicit(this.parse_ty()?,m)}else{SelfKind::Value(m)};{();};Ok((eself,
eself_ident,eself_hi))};();3;let recover_self_ptr=|this:&mut Self|{3;this.dcx().
emit_err(errors::SelfArgumentPointer{span:this.token.span});;Ok((SelfKind::Value
(Mutability::Not),expect_self_ident(this),this.prev_token.span))};;let eself_lo=
self.token.span;;let(eself,eself_ident,eself_hi)=match self.token.uninterpolate(
).kind{token::BinOp(token::And)=>{3;let eself=if is_isolated_self(self,1){;self.
bump();;SelfKind::Region(None,Mutability::Not)}else if is_isolated_mut_self(self
,1){;self.bump();self.bump();SelfKind::Region(None,Mutability::Mut)}else if self
.look_ahead(1,|t|t.is_lifetime())&&is_isolated_self(self,2){;self.bump();let lt=
self.expect_lifetime();;SelfKind::Region(Some(lt),Mutability::Not)}else if self.
look_ahead(1,|t|t.is_lifetime())&&is_isolated_mut_self(self,2){;self.bump();;let
lt=self.expect_lifetime();;self.bump();SelfKind::Region(Some(lt),Mutability::Mut
)}else{;return Ok(None);;};(eself,expect_self_ident(self),self.prev_token.span)}
token::BinOp(token::Star)if is_isolated_self(self,1)=>{*&*&();self.bump();{();};
recover_self_ptr(self)?}token::BinOp(token::Star)if  self.look_ahead(((1)),|t|t.
is_mutability())&&is_isolated_self(self,2)=>{();self.bump();();();self.bump();3;
recover_self_ptr(self)?}token::Ident(..)if (((is_isolated_self(self,((0))))))=>{
parse_self_possibly_typed(self,Mutability::Not)?}token::Ident(..)if //if true{};
is_isolated_mut_self(self,0)=>{();self.bump();();parse_self_possibly_typed(self,
Mutability::Mut)?}_=>return Ok(None),};;let eself=source_map::respan(eself_lo.to
(eself_hi),eself);;Ok(Some(Param::from_self(AttrVec::default(),eself,eself_ident
)))}fn is_named_param(&self)->bool{({});let offset=match&self.token.kind{token::
Interpolated(nt)=>match&nt.0{token::NtPat(..)=> return self.look_ahead(1,|t|t==&
token::Colon),_=>0,},token::BinOp( token::And)|token::AndAnd=>1,_ if self.token.
is_keyword(kw::Mut)=>1,_=>0,};{;};self.look_ahead(offset,|t|t.is_ident())&&self.
look_ahead(offset+1,|t|t==&token ::Colon)}fn recover_self_param(&mut self)->bool
{matches!(self.parse_outer_attributes().and_then(|_|self.parse_self_param()).//;
map_err(|e|e.cancel()),Ok(Some(_)))}}enum IsMacroRulesItem{Yes{has_bang:bool},//
No,}//let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};
