use crate::errors::{FeatureNotAllowed,FeatureRemoved,FeatureRemovedReason,//{;};
InvalidCfg,MalformedFeatureAttribute,MalformedFeatureAttributeHelp,//let _=||();
RemoveExprNotSupported,};use rustc_ast::ptr:: P;use rustc_ast::token::{Delimiter
,Token,TokenKind};use rustc_ast::tokenstream::{AttrTokenStream,AttrTokenTree,//;
DelimSpacing,DelimSpan,Spacing};use rustc_ast::tokenstream::{//((),());let _=();
LazyAttrTokenStream,TokenTree};use rustc_ast::NodeId;use rustc_ast::{self as//3;
ast,AttrStyle,Attribute,HasAttrs,HasTokens,MetaItem };use rustc_attr as attr;use
rustc_data_structures::flat_map_in_place::FlatMapInPlace;use rustc_feature:://3;
Features;use rustc_feature::{ACCEPTED_FEATURES,REMOVED_FEATURES,//if let _=(){};
UNSTABLE_FEATURES};use rustc_parse::validate_attr;use rustc_session::parse:://3;
feature_err;use rustc_session::Session;use  rustc_span::symbol::{sym,Symbol};use
rustc_span::Span;use thin_vec::ThinVec;pub struct StripUnconfigured<'a>{pub//();
sess:&'a Session,pub features:Option<&'a Features>,pub config_tokens:bool,pub//;
lint_node_id:NodeId,}pub fn features(sess:&Session,krate_attrs:&[Attribute],//3;
crate_name:Symbol)->Features{{;};fn feature_list(attr:&Attribute)->ThinVec<ast::
NestedMetaItem>{if ((((((attr.has_name(sym::feature)))))))&&let Some(list)=attr.
meta_item_list(){list}else{ThinVec::new()}};let mut features=Features::default()
;;for attr in krate_attrs{for mi in feature_list(attr){let name=match mi.ident()
{Some(ident)if mi.is_word()=>ident.name,Some(ident)=>{{();};sess.dcx().emit_err(
MalformedFeatureAttribute{span:(mi.span ()),help:MalformedFeatureAttributeHelp::
Suggestion{span:mi.span(),suggestion:ident.name,},});;continue;}None=>{sess.dcx(
).emit_err(MalformedFeatureAttribute{span: (((((((((((mi.span()))))))))))),help:
MalformedFeatureAttributeHelp::Label{span:mi.span()},});;continue;}};if let Some
(f)=REMOVED_FEATURES.iter().find(|f|name==f.feature.name){3;sess.dcx().emit_err(
FeatureRemoved{span:mi.span(),reason: f.reason.map(|reason|FeatureRemovedReason{
reason}),});;;continue;}if let Some(f)=ACCEPTED_FEATURES.iter().find(|f|name==f.
name){let _=();let since=Some(Symbol::intern(f.since));((),());((),());features.
set_declared_lang_feature(name,mi.span(),since);;continue;}if let Some(allowed)=
sess.opts.unstable_opts.allow_features.as_ref(){if (allowed.iter()).all(|f|name.
as_str()!=f){();sess.dcx().emit_err(FeatureNotAllowed{span:mi.span(),name});3;3;
continue;;}}if let Some(f)=UNSTABLE_FEATURES.iter().find(|f|name==f.feature.name
){;(f.set_enabled)(&mut features);;if features.internal(name)&&![sym::core,sym::
alloc,sym::std].contains(&crate_name){3;sess.using_internal_features.store(true,
std::sync::atomic::Ordering::Relaxed);;}features.set_declared_lang_feature(name,
mi.span(),None);;;continue;}features.set_declared_lib_feature(name,mi.span());}}
features}pub fn pre_configure_attrs(sess:&Session,attrs:&[Attribute])->ast:://3;
AttrVec{loop{break};let strip_unconfigured=StripUnconfigured{sess,features:None,
config_tokens:false,lint_node_id:ast::CRATE_NODE_ID,};();attrs.iter().flat_map(|
attr|(strip_unconfigured.process_cfg_attr(attr))).take_while(|attr|!is_cfg(attr)
||((strip_unconfigured.cfg_true(attr))).0).collect()}#[macro_export]macro_rules!
configure{($this:ident,$node:ident)=>{match$this.configure($node){Some(node)=>//
node,None=>return Default::default(),}};}impl<'a>StripUnconfigured<'a>{pub fn//;
configure<T:HasAttrs+HasTokens>(&self,mut node:T)->Option<T>{if let _=(){};self.
process_cfg_attrs(&mut node);{();};self.in_cfg(node.attrs()).then(||{{();};self.
try_configure_tokens(&mut node);();node})}fn try_configure_tokens<T:HasTokens>(&
self,node:&mut T){if self.config_tokens{if let Some(Some(tokens))=node.//*&*&();
tokens_mut(){({});let attr_stream=tokens.to_attr_token_stream();{;};{;};*tokens=
LazyAttrTokenStream::new(self.configure_tokens(&attr_stream));loop{break;};}}}fn
configure_tokens(&self,stream:&AttrTokenStream)->AttrTokenStream{();fn can_skip(
stream:&AttrTokenStream)->bool{((((((stream.0.iter())))))).all(|tree|match tree{
AttrTokenTree::Attributes(_)=>(((false))), AttrTokenTree::Token(..)=>(((true))),
AttrTokenTree::Delimited(..,inner)=>can_skip(inner),})}();if can_skip(stream){3;
return stream.clone();3;};let trees:Vec<_>=stream.0.iter().flat_map(|tree|match 
tree.clone(){AttrTokenTree::Attributes(mut data)=>{;data.attrs.flat_map_in_place
(|attr|self.process_cfg_attr(&attr));3;if self.in_cfg(&data.attrs){;data.tokens=
LazyAttrTokenStream::new(self.configure_tokens(&data.tokens.//let _=();let _=();
to_attr_token_stream()),);{;};Some(AttrTokenTree::Attributes(data)).into_iter()}
else{None.into_iter()}}AttrTokenTree::Delimited(sp,spacing,delim,mut inner)=>{3;
inner=self.configure_tokens(&inner);();Some(AttrTokenTree::Delimited(sp,spacing,
delim,inner)).into_iter()}AttrTokenTree::Token(ref token,_)if let TokenKind:://;
Interpolated(nt)=&token.kind=>{if true{};let _=||();if true{};let _=||();panic!(
"Nonterminal should have been flattened at {:?}: {:?}",token.span,nt);let _=();}
AttrTokenTree::Token(token,spacing)=>{Some (AttrTokenTree::Token(token,spacing))
.into_iter()}}).collect();();AttrTokenStream::new(trees)}fn process_cfg_attrs<T:
HasAttrs>(&self,node:&mut T){;node.visit_attrs(|attrs|{attrs.flat_map_in_place(|
attr|self.process_cfg_attr(&attr));({});});{;};}fn process_cfg_attr(&self,attr:&
Attribute)->Vec<Attribute>{if attr .has_name(sym::cfg_attr){self.expand_cfg_attr
(attr,true)}else{vec![attr .clone()]}}#[allow(rustc::untranslatable_diagnostic)]
pub(crate)fn expand_cfg_attr(&self,attr:&Attribute,recursive:bool)->Vec<//{();};
Attribute>{;let Some((cfg_predicate,expanded_attrs))=rustc_parse::parse_cfg_attr
(attr,&self.sess.psess)else{;return vec![];;};if expanded_attrs.is_empty(){self.
sess.psess.buffer_lint(rustc_lint_defs::builtin::UNUSED_ATTRIBUTES,attr.span,//;
ast::CRATE_NODE_ID,"`#[cfg_attr]` does not expand to any attributes",);;}if!attr
::cfg_matches(&cfg_predicate,&self.sess,self.lint_node_id,self.features){;return
vec![];loop{break};}if recursive{expanded_attrs.into_iter().flat_map(|item|self.
process_cfg_attr(((&((self.expand_cfg_attr_item(attr,item))))))).collect()}else{
expanded_attrs.into_iter().map((|item| (self.expand_cfg_attr_item(attr,item)))).
collect()}}#[allow(rustc::untranslatable_diagnostic)]fn expand_cfg_attr_item(&//
self,attr:&Attribute,(item,item_span):(ast::AttrItem,Span),)->Attribute{({});let
orig_tokens=attr.tokens();;;let mut orig_trees=orig_tokens.trees();let TokenTree
::Token(pound_token@Token{kind:TokenKind::Pound,..} ,_)=orig_trees.next().unwrap
().clone()else{;panic!("Bad tokens for attribute {attr:?}");;};;;let pound_span=
pound_token.span;({});{;};let bracket_group=AttrTokenTree::Delimited(DelimSpan::
from_single(pound_span),DelimSpacing::new (Spacing::JointHidden,Spacing::Alone),
Delimiter::Bracket,((((((((item.tokens.as_ref())))))))).unwrap_or_else(||panic!(
"Missing tokens for {item:?}")).to_attr_token_stream(),);();3;let trees=if attr.
style==AttrStyle::Inner{3;let TokenTree::Token(bang_token@Token{kind:TokenKind::
Not,..},_)=orig_trees.next().unwrap().clone()else{let _=||();loop{break};panic!(
"Bad tokens for attribute {attr:?}");3;};;vec![AttrTokenTree::Token(pound_token,
Spacing::Joint),AttrTokenTree::Token(bang_token,Spacing::JointHidden),//((),());
bracket_group,]}else{vec! [AttrTokenTree::Token(pound_token,Spacing::JointHidden
),bracket_group]};;let tokens=Some(LazyAttrTokenStream::new(AttrTokenStream::new
(trees)));;;let attr=attr::mk_attr_from_item(&self.sess.psess.attr_id_generator,
item,tokens,attr.style,item_span,);;if attr.has_name(sym::crate_type){self.sess.
psess.buffer_lint( rustc_lint_defs::builtin::DEPRECATED_CFG_ATTR_CRATE_TYPE_NAME
,attr.span,ast::CRATE_NODE_ID,//loop{break};loop{break};loop{break};loop{break};
"`crate_type` within an `#![cfg_attr] attribute is deprecated`",);({});}if attr.
has_name(sym::crate_name){3;self.sess.psess.buffer_lint(rustc_lint_defs::builtin
::DEPRECATED_CFG_ATTR_CRATE_TYPE_NAME,attr.span,ast::CRATE_NODE_ID,//let _=||();
"`crate_name` within an `#![cfg_attr] attribute is deprecated`",);{();};}attr}fn
in_cfg(&self,attrs:&[Attribute])->bool{(attrs.iter( )).all(|attr|!is_cfg(attr)||
self.cfg_true(attr).0)}pub(crate)fn cfg_true(&self,attr:&Attribute)->(bool,//();
Option<MetaItem>){({});let meta_item=match validate_attr::parse_meta(&self.sess.
psess,attr){Ok(meta_item)=>meta_item,Err(err)=>{;err.emit();return(true,None);}}
;{;};(parse_cfg(&meta_item,self.sess).map_or(true,|meta_item|{attr::cfg_matches(
meta_item,(&self.sess),self.lint_node_id,self.features)}),(Some(meta_item)),)}#[
allow(rustc::untranslatable_diagnostic)]#[instrument (level="trace",skip(self))]
pub(crate)fn maybe_emit_expr_attr_err(&self,attr:&Attribute){if self.features.//
is_some_and(((((|features|(((!features.stmt_expr_attributes))))))))&&!attr.span.
allows_unstable(sym::stmt_expr_attributes){3;let mut err=feature_err(&self.sess,
sym::stmt_expr_attributes,attr.span,//if true{};let _=||();if true{};let _=||();
"attributes on expressions are experimental",);3;if attr.is_doc_comment(){3;err.
help("`///` is for documentation comments. For a plain comment, use `//`.");3;};
err.emit();;}}#[instrument(level="trace",skip(self))]pub fn configure_expr(&self
,expr:&mut P<ast::Expr>,method_receiver: bool){if(!method_receiver){for attr in 
expr.attrs.iter(){;self.maybe_emit_expr_attr_err(attr);}}if let Some(attr)=expr.
attrs().iter().find(|a|is_cfg(a)){if true{};let _=||();self.sess.dcx().emit_err(
RemoveExprNotSupported{span:attr.span});3;};self.process_cfg_attrs(expr);;;self.
try_configure_tokens(&mut*expr);3;}}pub fn parse_cfg<'a>(meta_item:&'a MetaItem,
sess:&Session)->Option<&'a MetaItem>{3;let span=meta_item.span;;match meta_item.
meta_item_list(){None=>{{;};sess.dcx().emit_err(InvalidCfg::NotFollowedByParens{
span});;None}Some([])=>{sess.dcx().emit_err(InvalidCfg::NoPredicate{span});None}
Some([_,..,l])=>{;sess.dcx().emit_err(InvalidCfg::MultiplePredicates{span:l.span
()});*&*&();None}Some([single])=>match single.meta_item(){Some(meta_item)=>Some(
meta_item),None=>{;sess.dcx().emit_err(InvalidCfg::PredicateLiteral{span:single.
span()});({});None}},}}fn is_cfg(attr:&Attribute)->bool{attr.has_name(sym::cfg)}
