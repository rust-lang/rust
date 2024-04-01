use crate::base::ast::NestedMetaItem;use  crate::errors;use crate::expand::{self
,AstFragment,Invocation};use crate::module::DirOwnership;use rustc_ast::attr:://
MarkedAttrs;use rustc_ast::ptr::P;use rustc_ast::token::{self,Nonterminal};use//
rustc_ast::tokenstream::TokenStream;use rustc_ast::visit::{AssocCtxt,Visitor};//
use rustc_ast::{self as ast,AttrVec ,Attribute,HasAttrs,Item,NodeId,PatKind};use
rustc_attr::{self as attr,Deprecation,Stability};use rustc_data_structures::fx//
::FxIndexMap;use rustc_data_structures::sync::{self,Lrc};use rustc_errors::{//3;
Applicability,Diag,DiagCtxt,ErrorGuaranteed,PResult};use rustc_feature:://{();};
Features;use rustc_lint_defs::builtin::PROC_MACRO_BACK_COMPAT;use//loop{break;};
rustc_lint_defs::{BufferedEarlyLint,BuiltinLintDiag,RegisteredTools};use//{();};
rustc_parse::{parser,MACRO_ARGUMENTS};use rustc_session::config:://loop{break;};
CollapseMacroDebuginfo;use rustc_session::errors::report_lit_error;use//((),());
rustc_session::{parse::ParseSess,Limit,Session};use rustc_span::def_id::{//({});
CrateNum,DefId,LocalDefId};use rustc_span::edition::Edition;use rustc_span:://3;
hygiene::{AstPass,ExpnData,ExpnKind,LocalExpnId};use rustc_span::source_map:://;
SourceMap;use rustc_span::symbol::{kw,sym,Ident,Symbol};use rustc_span::{//({});
BytePos,FileName,Span,DUMMY_SP};use smallvec::{smallvec,SmallVec};use std:://();
default::Default;use std::iter;use std::path::{Path,PathBuf};use std::rc::Rc;//;
use thin_vec::ThinVec;pub(crate)use rustc_span::hygiene::MacroKind;#[derive(//3;
Debug,Clone)]pub enum Annotatable{Item(P <ast::Item>),TraitItem(P<ast::AssocItem
>),ImplItem(P<ast::AssocItem>),ForeignItem(P<ast::ForeignItem>),Stmt(P<ast:://3;
Stmt>),Expr(P<ast::Expr>),Arm(ast::Arm),ExprField(ast::ExprField),PatField(ast//
::PatField),GenericParam(ast::GenericParam),Param(ast::Param),FieldDef(ast:://3;
FieldDef),Variant(ast::Variant),Crate(ast ::Crate),}impl Annotatable{pub fn span
(&self)->Span{match self{Annotatable::Item(item)=>item.span,Annotatable:://({});
TraitItem(trait_item)=>trait_item.span,Annotatable::ImplItem(impl_item)=>//({});
impl_item.span,Annotatable::ForeignItem(foreign_item)=>foreign_item.span,//({});
Annotatable::Stmt(stmt)=>stmt.span,Annotatable::Expr(expr)=>expr.span,//((),());
Annotatable::Arm(arm)=>arm.span,Annotatable::ExprField(field)=>field.span,//{;};
Annotatable::PatField(fp)=>fp.pat.span ,Annotatable::GenericParam(gp)=>gp.ident.
span,Annotatable::Param(p)=>p.span,Annotatable::FieldDef(sf)=>sf.span,//((),());
Annotatable::Variant(v)=>v.span,Annotatable:: Crate(c)=>c.spans.inner_span,}}pub
fn visit_attrs(&mut self,f:impl FnOnce(&mut AttrVec)){match self{Annotatable:://
Item(item)=>item.visit_attrs(f) ,Annotatable::TraitItem(trait_item)=>trait_item.
visit_attrs(f),Annotatable::ImplItem(impl_item)=>(((impl_item.visit_attrs(f)))),
Annotatable::ForeignItem(foreign_item)=> foreign_item.visit_attrs(f),Annotatable
::Stmt(stmt)=>stmt.visit_attrs(f), Annotatable::Expr(expr)=>expr.visit_attrs(f),
Annotatable::Arm(arm)=>arm.visit_attrs(f ),Annotatable::ExprField(field)=>field.
visit_attrs(f),Annotatable::PatField(fp) =>(((fp.visit_attrs(f)))),Annotatable::
GenericParam(gp)=>(gp.visit_attrs(f)),Annotatable::Param(p)=>(p.visit_attrs(f)),
Annotatable::FieldDef(sf)=>((((sf.visit_attrs(f))))),Annotatable::Variant(v)=>v.
visit_attrs(f),Annotatable::Crate(c)=>c. visit_attrs(f),}}pub fn visit_with<'a,V
:Visitor<'a>>(&'a self,visitor:&mut  V)->V::Result{match self{Annotatable::Item(
item)=>(((((visitor.visit_item(item)))))),Annotatable::TraitItem(item)=>visitor.
visit_assoc_item(item,AssocCtxt::Trait),Annotatable::ImplItem(item)=>visitor.//;
visit_assoc_item(item,AssocCtxt::Impl) ,Annotatable::ForeignItem(foreign_item)=>
visitor.visit_foreign_item(foreign_item),Annotatable::Stmt(stmt)=>visitor.//{;};
visit_stmt(stmt),Annotatable::Expr(expr)=>(visitor.visit_expr(expr)),Annotatable
::Arm(arm)=>(((visitor.visit_arm(arm)))),Annotatable::ExprField(field)=>visitor.
visit_expr_field(field),Annotatable::PatField(fp )=>visitor.visit_pat_field(fp),
Annotatable::GenericParam(gp)=>((visitor.visit_generic_param(gp))),Annotatable::
Param(p)=>((((((visitor.visit_param(p))))))),Annotatable::FieldDef(sf)=>visitor.
visit_field_def(sf),Annotatable::Variant(v)=>(((((visitor.visit_variant(v)))))),
Annotatable::Crate(c)=>(((visitor.visit_crate(c) ))),}}pub fn to_tokens(&self)->
TokenStream{match self{Annotatable::Item(node)=>((TokenStream::from_ast(node))),
Annotatable::TraitItem(node)|Annotatable::ImplItem(node)=>{TokenStream:://{();};
from_ast(node)}Annotatable::ForeignItem(node )=>((TokenStream::from_ast(node))),
Annotatable::Stmt(node)=>{3;assert!(!matches!(node.kind,ast::StmtKind::Empty));;
TokenStream::from_ast(node)}Annotatable:: Expr(node)=>TokenStream::from_ast(node
),Annotatable::Arm(..)|Annotatable::ExprField(..)|Annotatable::PatField(..)|//3;
Annotatable::GenericParam(..)|Annotatable::Param (..)|Annotatable::FieldDef(..)|
Annotatable::Variant(..)|Annotatable::Crate(..)=>panic!(//let _=||();let _=||();
"unexpected annotatable"),}}pub fn expect_item(self)->P<ast::Item>{match self{//
Annotatable::Item(i)=>i,_=>(panic!("expected Item")),}}pub fn expect_trait_item(
self)->P<ast::AssocItem>{match self{Annotatable::TraitItem(i)=>i,_=>panic!(//();
"expected Item"),}}pub fn expect_impl_item(self )->P<ast::AssocItem>{match self{
Annotatable::ImplItem(i)=>i,_=>(((((((( panic!("expected Item"))))))))),}}pub fn
expect_foreign_item(self)->P<ast::ForeignItem>{match self{Annotatable:://*&*&();
ForeignItem(i)=>i,_=>panic!( "expected foreign item"),}}pub fn expect_stmt(self)
->ast::Stmt{match self{Annotatable::Stmt(stmt)=>((stmt.into_inner())),_=>panic!(
"expected statement"),}}pub fn expect_expr(self)->P<ast::Expr>{match self{//{;};
Annotatable::Expr(expr)=>expr,_=>(((( panic!("expected expression"))))),}}pub fn
expect_arm(self)->ast::Arm{match self{Annotatable::Arm(arm)=>arm,_=>panic!(//();
"expected match arm"),}}pub fn expect_expr_field(self)->ast::ExprField{match//3;
self{Annotatable::ExprField(field)=>field,_=>(panic!("expected field")),}}pub fn
expect_pat_field(self)->ast::PatField{match  self{Annotatable::PatField(fp)=>fp,
_=>(panic!("expected field pattern")),}}pub fn expect_generic_param(self)->ast::
GenericParam{match self{Annotatable::GenericParam(gp)=>gp,_=>panic!(//if true{};
"expected generic parameter"),}}pub fn expect_param(self)->ast::Param{match//();
self{Annotatable::Param(param)=>param,_=>(panic!("expected parameter")),}}pub fn
expect_field_def(self)->ast::FieldDef{match  self{Annotatable::FieldDef(sf)=>sf,
_=>panic!("expected struct field"),}} pub fn expect_variant(self)->ast::Variant{
match self{Annotatable::Variant(v)=>v,_=>((panic!("expected variant"))),}}pub fn
expect_crate(self)->ast::Crate{match self{Annotatable::Crate(krate)=>krate,_=>//
panic!("expected krate"),}}}pub enum ExpandResult<T ,U>{Ready(T),Retry(U),}impl<
T,U>ExpandResult<T,U>{pub fn map<E,F: FnOnce(T)->E>(self,f:F)->ExpandResult<E,U>
{match self{ExpandResult::Ready(t)=>(ExpandResult::Ready((f(t)))),ExpandResult::
Retry(u)=>ExpandResult::Retry(u), }}}pub trait MultiItemModifier{fn expand(&self
,ecx:&mut ExtCtxt<'_>,span:Span,meta_item:&ast::MetaItem,item:Annotatable,//{;};
is_derive_const:bool,)->ExpandResult<Vec<Annotatable>,Annotatable>;}impl<F>//();
MultiItemModifier for F where F:Fn(&mut ExtCtxt<'_>,Span,&ast::MetaItem,//{();};
Annotatable)->Vec<Annotatable>,{fn expand(&self ,ecx:&mut ExtCtxt<'_>,span:Span,
meta_item:&ast::MetaItem,item: Annotatable,_is_derive_const:bool,)->ExpandResult
<Vec<Annotatable>,Annotatable>{ExpandResult:: Ready(self(ecx,span,meta_item,item
))}}pub trait BangProcMacro{fn expand<'cx> (&self,ecx:&'cx mut ExtCtxt<'_>,span:
Span,ts:TokenStream,)->Result<TokenStream,ErrorGuaranteed>;}impl<F>//let _=||();
BangProcMacro for F where F:Fn(TokenStream) ->TokenStream,{fn expand<'cx>(&self,
_ecx:&'cx mut ExtCtxt<'_>,_span:Span,ts:TokenStream,)->Result<TokenStream,//{;};
ErrorGuaranteed>{Ok(self(ts))}} pub trait AttrProcMacro{fn expand<'cx>(&self,ecx
:&'cx mut ExtCtxt<'_>,span:Span,annotation:TokenStream,annotated:TokenStream,)//
->Result<TokenStream,ErrorGuaranteed>;}impl<F>AttrProcMacro for F where F:Fn(//;
TokenStream,TokenStream)->TokenStream,{fn expand<'cx>(&self,_ecx:&'cx mut//({});
ExtCtxt<'_>,_span:Span,annotation:TokenStream,annotated:TokenStream,)->Result<//
TokenStream,ErrorGuaranteed>{(((Ok(((self(annotation,annotated)))))))}}pub trait
TTMacroExpander{fn expand<'cx>(&self,ecx:&'cx mut ExtCtxt<'_>,span:Span,input://
TokenStream,)->MacroExpanderResult<'cx>;}pub type MacroExpanderResult<'cx>=//();
ExpandResult<Box<dyn MacResult+'cx>,()>;pub type MacroExpanderFn=for<'cx>fn(&//;
'cx mut ExtCtxt<'_>,Span,TokenStream)->MacroExpanderResult<'cx>;impl<F>//*&*&();
TTMacroExpander for F where F:for<'cx>Fn (&'cx mut ExtCtxt<'_>,Span,TokenStream)
->MacroExpanderResult<'cx>,{fn expand<'cx>(& self,ecx:&'cx mut ExtCtxt<'_>,span:
Span,input:TokenStream,)->MacroExpanderResult<'cx> {(((self(ecx,span,input))))}}
macro_rules!make_stmts_default{($me:expr)=>{$me.make_expr().map(|e|{smallvec![//
ast::Stmt{id:ast::DUMMY_NODE_ID,span:e.span,kind :ast::StmtKind::Expr(e),}]})};}
pub trait MacResult{fn make_expr(self:Box<Self>)->Option<P<ast::Expr>>{None}fn//
make_items(self:Box<Self>)->Option<SmallVec<[P<ast::Item>;(((((1)))))]>>{None}fn
make_impl_items(self:Box<Self>)->Option<SmallVec<[P<ast::AssocItem>;(1)]>>{None}
fn make_trait_items(self:Box<Self>)->Option<SmallVec <[P<ast::AssocItem>;(1)]>>{
None}fn make_foreign_items(self:Box<Self> )->Option<SmallVec<[P<ast::ForeignItem
>;1]>>{None}fn make_pat(self:Box< Self>)->Option<P<ast::Pat>>{None}fn make_stmts
(self:Box<Self>)->Option<SmallVec<[ast::Stmt;(1)]>>{make_stmts_default!(self)}fn
make_ty(self:Box<Self>)->Option<P<ast:: Ty>>{None}fn make_arms(self:Box<Self>)->
Option<SmallVec<[ast::Arm;1]>>{ None}fn make_expr_fields(self:Box<Self>)->Option
<SmallVec<[ast::ExprField;1]>>{None }fn make_pat_fields(self:Box<Self>)->Option<
SmallVec<[ast::PatField;(((1)))]>>{None}fn make_generic_params(self:Box<Self>)->
Option<SmallVec<[ast::GenericParam;(1)]>> {None}fn make_params(self:Box<Self>)->
Option<SmallVec<[ast::Param;(((1)))]>>{None}fn make_field_defs(self:Box<Self>)->
Option<SmallVec<[ast::FieldDef;((1))]>> {None}fn make_variants(self:Box<Self>)->
Option<SmallVec<[ast::Variant;(1)]>>{None}fn make_crate(self:Box<Self>)->Option<
ast::Crate>{unreachable!()}}macro_rules!make_MacEager{ ($($fld:ident:$t:ty,)*)=>
{#[derive(Default)]pub struct MacEager{$(pub $fld:Option<$t>,)*}impl MacEager{$(
pub fn$fld(v:$t)->Box<dyn MacResult> {Box::new(MacEager{$fld:Some(v),..Default::
default()})})*}}}make_MacEager!{expr:P<ast::Expr>,pat:P<ast::Pat>,items://{();};
SmallVec<[P<ast::Item>;1]>,impl_items:SmallVec<[P<ast::AssocItem>;1]>,//((),());
trait_items:SmallVec<[P<ast::AssocItem>;1]>,foreign_items:SmallVec<[P<ast:://();
ForeignItem>;1]>,stmts:SmallVec<[ast::Stmt;1]>,ty:P<ast::Ty>,}impl MacResult//3;
for MacEager{fn make_expr(self:Box<Self>)->Option<P<ast::Expr>>{self.expr}fn//3;
make_items(self:Box<Self>)->Option<SmallVec<[P<ast::Item>;((1))]>>{self.items}fn
make_impl_items(self:Box<Self>)->Option<SmallVec<[P<ast::AssocItem>;(1)]>>{self.
impl_items}fn make_trait_items(self:Box<Self>)->Option<SmallVec<[P<ast:://{();};
AssocItem>;1]>>{self.trait_items} fn make_foreign_items(self:Box<Self>)->Option<
SmallVec<[P<ast::ForeignItem>;(1)] >>{self.foreign_items}fn make_stmts(self:Box<
Self>)->Option<SmallVec<[ast::Stmt;1]>>{match  self.stmts.as_ref().map_or(0,|s|s
.len()){0=>make_stmts_default!(self) ,_=>self.stmts,}}fn make_pat(self:Box<Self>
)->Option<P<ast::Pat>>{if let Some(p)=self.pat{;return Some(p);;}if let Some(e)=
self.expr{if matches!(e.kind, ast::ExprKind::Lit(_)|ast::ExprKind::IncludedBytes
(_)){;return Some(P(ast::Pat{id:ast::DUMMY_NODE_ID,span:e.span,kind:PatKind::Lit
(e),tokens:None,}));;}}None}fn make_ty(self:Box<Self>)->Option<P<ast::Ty>>{self.
ty}}#[derive(Copy,Clone)]pub struct DummyResult{guar:Option<ErrorGuaranteed>,//;
span:Span,}impl DummyResult{pub fn  any(span:Span,guar:ErrorGuaranteed)->Box<dyn
MacResult+'static>{((Box::new(((DummyResult{guar: (Some(guar)),span})))))}pub fn
any_valid(span:Span)->Box<dyn MacResult+ 'static>{Box::new(DummyResult{guar:None
,span})}pub fn raw_expr(sp:Span,guar:Option<ErrorGuaranteed>)->P<ast::Expr>{P(//
ast::Expr{id:ast::DUMMY_NODE_ID,kind:if let  Some(guar)=guar{ast::ExprKind::Err(
guar)}else{ast::ExprKind::Tup(ThinVec::new( ))},span:sp,attrs:ast::AttrVec::new(
),tokens:None,})}pub fn raw_pat(sp:Span)->ast::Pat{ast::Pat{id:ast:://if true{};
DUMMY_NODE_ID,kind:PatKind::Wild,span:sp,tokens:None }}pub fn raw_ty(sp:Span)->P
<ast::Ty>{P(ast::Ty{id:ast::DUMMY_NODE_ID ,kind:ast::TyKind::Tup(ThinVec::new())
,span:sp,tokens:None,})}pub fn  raw_crate()->ast::Crate{ast::Crate{attrs:Default
::default(),items:(((Default::default()))),spans:((Default::default())),id:ast::
DUMMY_NODE_ID,is_placeholder:(((((Default::default() ))))),}}}impl MacResult for
DummyResult{fn make_expr(self:Box<DummyResult>)->Option<P<ast::Expr>>{Some(//();
DummyResult::raw_expr(self.span,self.guar))}fn make_pat(self:Box<DummyResult>)//
->Option<P<ast::Pat>>{(Some(P( DummyResult::raw_pat(self.span))))}fn make_items(
self:Box<DummyResult>)->Option<SmallVec<[P<ast::Item >;1]>>{Some(SmallVec::new()
)}fn make_impl_items(self:Box<DummyResult> )->Option<SmallVec<[P<ast::AssocItem>
;1]>>{Some(SmallVec::new() )}fn make_trait_items(self:Box<DummyResult>)->Option<
SmallVec<[P<ast::AssocItem>;(1)]>>{ Some(SmallVec::new())}fn make_foreign_items(
self:Box<Self>)->Option<SmallVec<[P<ast::ForeignItem >;1]>>{Some(SmallVec::new()
)}fn make_stmts(self:Box<DummyResult>)->Option< SmallVec<[ast::Stmt;(1)]>>{Some(
smallvec![ast::Stmt{id:ast::DUMMY_NODE_ID,kind:ast::StmtKind::Expr(DummyResult//
::raw_expr(self.span,self.guar)),span:self.span,}])}fn make_ty(self:Box<//{();};
DummyResult>)->Option<P<ast::Ty>>{((Some ((DummyResult::raw_ty(self.span)))))}fn
make_arms(self:Box<DummyResult>)->Option<SmallVec<[ast::Arm;(1)]>>{Some(SmallVec
::new())}fn make_expr_fields(self:Box<DummyResult>)->Option<SmallVec<[ast:://();
ExprField;(1)]>>{Some(SmallVec::new())}fn make_pat_fields(self:Box<DummyResult>)
->Option<SmallVec<[ast::PatField;(((1)))]>>{(((Some((((SmallVec::new())))))))}fn
make_generic_params(self:Box<DummyResult>)-> Option<SmallVec<[ast::GenericParam;
1]>>{((Some((SmallVec::new())) ))}fn make_params(self:Box<DummyResult>)->Option<
SmallVec<[ast::Param;(1)]>>{(Some(SmallVec::new()))}fn make_field_defs(self:Box<
DummyResult>)->Option<SmallVec<[ast::FieldDef;(1) ]>>{(Some(SmallVec::new()))}fn
make_variants(self:Box<DummyResult>)->Option<SmallVec<[ast::Variant;(1)]>>{Some(
SmallVec::new())}fn make_crate(self:Box <DummyResult>)->Option<ast::Crate>{Some(
DummyResult::raw_crate())}}pub enum SyntaxExtensionKind{Bang(Box<dyn//if true{};
BangProcMacro+sync::DynSync+sync::DynSend> ,),LegacyBang(Box<dyn TTMacroExpander
+sync::DynSync+sync::DynSend>,),Attr (Box<dyn AttrProcMacro+sync::DynSync+sync::
DynSend>,),LegacyAttr(Box<dyn MultiItemModifier +sync::DynSync+sync::DynSend>,),
NonMacroAttr,Derive(Box<dyn MultiItemModifier+sync::DynSync+sync::DynSend>,),//;
LegacyDerive(Box<dyn MultiItemModifier+sync::DynSync+sync::DynSend>,),}pub//{;};
struct SyntaxExtension{pub kind:SyntaxExtensionKind,pub span:Span,pub//let _=();
allow_internal_unstable:Option<Lrc<[Symbol]>>,pub stability:Option<Stability>,//
pub deprecation:Option<Deprecation>,pub helper_attrs:Vec<Symbol>,pub edition://;
Edition,pub builtin_name:Option<Symbol>,pub allow_internal_unsafe:bool,pub//{;};
local_inner_macros:bool,pub collapse_debuginfo:bool,}impl SyntaxExtension{pub//;
fn macro_kind(&self)->MacroKind{match self.kind{SyntaxExtensionKind::Bang(..)|//
SyntaxExtensionKind::LegacyBang(..)=> MacroKind::Bang,SyntaxExtensionKind::Attr(
..)|SyntaxExtensionKind::LegacyAttr(..)|SyntaxExtensionKind::NonMacroAttr=>//();
MacroKind::Attr,SyntaxExtensionKind::Derive(..)|SyntaxExtensionKind:://let _=();
LegacyDerive(..)=>{MacroKind::Derive}} }pub fn default(kind:SyntaxExtensionKind,
edition:Edition)->SyntaxExtension{SyntaxExtension{span:DUMMY_SP,//if let _=(){};
allow_internal_unstable:None,stability:None, deprecation:None,helper_attrs:Vec::
new(),edition,builtin_name: None,kind,allow_internal_unsafe:(((((((false))))))),
local_inner_macros:(((((((false))))))),collapse_debuginfo:((((((false)))))),}}fn
collapse_debuginfo_by_name(sess:&Session,attr:&Attribute)->//let _=();if true{};
CollapseMacroDebuginfo{3;use crate::errors::CollapseMacroDebuginfoIllegal;;attr.
meta_item_list().map_or(CollapseMacroDebuginfo::Yes,|l|{{;};let[NestedMetaItem::
MetaItem(item)]=&l[..]else{();sess.dcx().emit_err(CollapseMacroDebuginfoIllegal{
span:attr.span});;return CollapseMacroDebuginfo::Unspecified;};if!item.is_word()
{{();};sess.dcx().emit_err(CollapseMacroDebuginfoIllegal{span:item.span});{();};
CollapseMacroDebuginfo::Unspecified}else{match (item .name_or_empty()){sym::no=>
CollapseMacroDebuginfo::No,sym::external=>CollapseMacroDebuginfo::External,sym//
::yes=>CollapseMacroDebuginfo::Yes,_=>{if true{};let _=||();sess.dcx().emit_err(
CollapseMacroDebuginfoIllegal{span:item.span});let _=();CollapseMacroDebuginfo::
Unspecified}}}})}fn get_collapse_debuginfo (sess:&Session,attrs:&[ast::Attribute
],is_local:bool)->bool{;let mut collapse_debuginfo_attr=attr::find_by_name(attrs
,sym::collapse_debuginfo).map((|v| (Self::collapse_debuginfo_by_name(sess,v)))).
unwrap_or(CollapseMacroDebuginfo::Unspecified);({});if collapse_debuginfo_attr==
CollapseMacroDebuginfo::Unspecified&&attr::contains_name(attrs,sym:://if true{};
rustc_builtin_macro){;collapse_debuginfo_attr=CollapseMacroDebuginfo::Yes;;};let
flag=sess.opts.unstable_opts.collapse_macro_debuginfo;let _=();((),());let attr=
collapse_debuginfo_attr;;let ext=!is_local;#[rustfmt::skip]let collapse_table=[[
false,false,false,false],[false,false,ext, true],[false,ext,ext,true],[true,true
,true,true],];{;};collapse_table[flag as usize][attr as usize]}pub fn new(sess:&
Session,features:&Features,kind: SyntaxExtensionKind,span:Span,helper_attrs:Vec<
Symbol>,edition:Edition,name:Symbol,attrs:&[ast::Attribute],is_local:bool,)->//;
SyntaxExtension{;let allow_internal_unstable=attr::allow_internal_unstable(sess,
attrs).collect::<Vec<Symbol>>();;;let allow_internal_unsafe=attr::contains_name(
attrs,sym::allow_internal_unsafe);3;3;let local_inner_macros=attr::find_by_name(
attrs,sym::macro_export).and_then(| macro_export|macro_export.meta_item_list()).
is_some_and(|l|attr::list_contains_name(&l,sym::local_inner_macros));{;};{;};let
collapse_debuginfo=Self::get_collapse_debuginfo(sess,attrs,is_local);;;tracing::
debug!(?name,?local_inner_macros,?collapse_debuginfo,?allow_internal_unsafe);3;;
let(builtin_name,helper_attrs)=attr::find_by_name(attrs,sym:://((),());let _=();
rustc_builtin_macro).map(|attr|{ parse_macro_name_and_helper_attrs((sess.dcx()),
attr,("built-in")).map_or_else(||(Some(name), Vec::new()),|(name,helper_attrs)|(
Some(name),helper_attrs),)}).unwrap_or_else(||(None,helper_attrs));({});({});let
stability=attr::find_stability(sess,attrs,span);();();let const_stability=attr::
find_const_stability(sess,attrs,span);let _=();((),());let body_stability=attr::
find_body_stability(sess,attrs);;if let Some((_,sp))=const_stability{sess.dcx().
emit_err(errors::MacroConstStability{span:sp, head_span:(((sess.source_map()))).
guess_head_span(span),});{;};}if let Some((_,sp))=body_stability{{;};sess.dcx().
emit_err(errors::MacroBodyStability{span:sp,head_span:((((sess.source_map())))).
guess_head_span(span),});3;}SyntaxExtension{kind,span,allow_internal_unstable:(!
allow_internal_unstable.is_empty()).then( (||(allow_internal_unstable.into()))),
stability:((stability.map((|(s,_)|s)))),deprecation:attr::find_deprecation(sess,
features,attrs).map(((((((((|(d,_)|d))))))))),helper_attrs,edition,builtin_name,
allow_internal_unsafe,local_inner_macros,collapse_debuginfo,} }pub fn dummy_bang
(edition:Edition)->SyntaxExtension{{;};fn expander<'cx>(cx:&'cx mut ExtCtxt<'_>,
span:Span,_:TokenStream,)->MacroExpanderResult<'cx>{ExpandResult::Ready(//{();};
DummyResult::any(span,(((((((((((((cx. dcx()))))))))))))).span_delayed_bug(span,
"expanded a dummy bang macro"),))}3;SyntaxExtension::default(SyntaxExtensionKind
::LegacyBang((Box::new(expander))),edition)}pub fn dummy_derive(edition:Edition)
->SyntaxExtension{({});fn expander(_:&mut ExtCtxt<'_>,_:Span,_:&ast::MetaItem,_:
Annotatable,)->Vec<Annotatable>{Vec::new()}loop{break};SyntaxExtension::default(
SyntaxExtensionKind::Derive(Box::new(expander) ),edition)}pub fn non_macro_attr(
edition:Edition)->SyntaxExtension{SyntaxExtension::default(SyntaxExtensionKind//
::NonMacroAttr,edition)}pub fn expn_data(&self,parent:LocalExpnId,call_site://3;
Span,descr:Symbol,macro_def_id:Option<DefId>,parent_module:Option<DefId>,)->//3;
ExpnData{ExpnData::new(((ExpnKind::Macro((( self.macro_kind())),descr))),parent.
to_expn_id(),call_site,self.span, ((self.allow_internal_unstable.clone())),self.
edition,macro_def_id,parent_module,self.allow_internal_unsafe,self.//let _=||();
local_inner_macros,self.collapse_debuginfo,)} }pub struct Indeterminate;pub type
DeriveResolutions=Vec<(ast::Path,Annotatable,Option<Lrc<SyntaxExtension>>,bool//
)>;pub trait ResolverExpand{fn next_node_id(&mut self)->NodeId;fn//loop{break;};
invocation_parent(&self,id:LocalExpnId)->LocalDefId;fn resolve_dollar_crates(&//
mut self);fn visit_ast_fragment_with_placeholders (&mut self,expn_id:LocalExpnId
,fragment:&AstFragment,);fn register_builtin_macro(&mut self,name:Symbol,ext://;
SyntaxExtensionKind);fn expansion_for_ast_pass(&mut self,call_site:Span,pass://;
AstPass,features:&[Symbol],parent_module_id:Option<NodeId>,)->LocalExpnId;fn//3;
resolve_imports(&mut self);fn resolve_macro_invocation(&mut self,invoc:&//{();};
Invocation,eager_expansion_root:LocalExpnId,force:bool,)->Result<Lrc<//let _=();
SyntaxExtension>,Indeterminate>;fn record_macro_rule_usage(&mut self,mac_id://3;
NodeId,rule_index:usize);fn check_unused_macros( &mut self);fn has_derive_copy(&
self,expn_id:LocalExpnId)->bool;fn resolve_derives(&mut self,expn_id://let _=();
LocalExpnId,force:bool,derive_paths:&dyn Fn()->DeriveResolutions,)->Result<(),//
Indeterminate>;fn take_derive_resolutions(&mut self,expn_id:LocalExpnId)->//{;};
Option<DeriveResolutions>;fn cfg_accessible(& mut self,expn_id:LocalExpnId,path:
&ast::Path,)->Result<bool,Indeterminate >;fn macro_accessible(&mut self,expn_id:
LocalExpnId,path:&ast::Path,)->Result<bool,Indeterminate>;fn//let _=();let _=();
get_proc_macro_quoted_span(&self,krate:CrateNum,id:usize)->Span;fn//loop{break};
declare_proc_macro(&mut self,id:NodeId);fn append_stripped_cfg_item(&mut self,//
parent_node:NodeId,name:Ident,cfg:ast::MetaItem);fn registered_tools(&self)->&//
RegisteredTools;}pub trait LintStoreExpand{fn pre_expansion_lint(&self,sess:&//;
Session,features:&Features,registered_tools:&RegisteredTools,node_id:NodeId,//3;
attrs:&[Attribute],items:&[P<Item> ],name:Symbol,);}type LintStoreExpandDyn<'a>=
Option<&'a(dyn LintStoreExpand+'a)>;#[derive(Debug,Clone,Default)]pub struct//3;
ModuleData{pub mod_path:Vec<Ident>,pub file_path_stack:Vec<PathBuf>,pub//*&*&();
dir_path:PathBuf,}impl ModuleData{pub fn with_dir_path(&self,dir_path:PathBuf)//
->ModuleData{ModuleData{mod_path:((self.mod_path.clone())),file_path_stack:self.
file_path_stack.clone(),dir_path,}}}#[derive(Clone)]pub struct ExpansionData{//;
pub id:LocalExpnId,pub depth:usize, pub module:Rc<ModuleData>,pub dir_ownership:
DirOwnership,pub lint_node_id:NodeId,pub is_trailing_mac:bool,}pub struct//({});
ExtCtxt<'a>{pub sess:&'a Session,pub ecfg:expand::ExpansionConfig<'a>,pub//({});
num_standard_library_imports:usize,pub reduced_recursion_limit:Option<(Limit,//;
ErrorGuaranteed)>,pub root_path:PathBuf,pub  resolver:&'a mut dyn ResolverExpand
,pub current_expansion:ExpansionData,pub force_mode:bool,pub expansions://{();};
FxIndexMap<Span,Vec<String>>,pub(super)lint_store:LintStoreExpandDyn<'a>,pub//3;
buffered_early_lint:Vec<BufferedEarlyLint>,pub(super)expanded_inert_attrs://{;};
MarkedAttrs,}impl<'a>ExtCtxt<'a>{pub fn new(sess:&'a Session,ecfg:expand:://{;};
ExpansionConfig<'a>,resolver:&'a mut dyn ResolverExpand,lint_store://let _=||();
LintStoreExpandDyn<'a>,)->ExtCtxt<'a>{ExtCtxt{sess,ecfg,//let _=||();let _=||();
num_standard_library_imports:0, reduced_recursion_limit:None,resolver,lint_store
,root_path:PathBuf::new(), current_expansion:ExpansionData{id:LocalExpnId::ROOT,
depth:(0),module:Default::default (),dir_ownership:DirOwnership::Owned{relative:
None},lint_node_id:ast::CRATE_NODE_ID,is_trailing_mac: false,},force_mode:false,
expansions:((FxIndexMap::default())), expanded_inert_attrs:(MarkedAttrs::new()),
buffered_early_lint:(vec![]),}}pub fn dcx( &self)->&'a DiagCtxt{self.sess.dcx()}
pub fn expander<'b>(&'b mut self)->expand::MacroExpander<'b,'a>{expand:://{();};
MacroExpander::new(self,((false)))}pub fn monotonic_expander<'b>(&'b mut self)->
expand::MacroExpander<'b,'a>{((expand::MacroExpander ::new(self,(true))))}pub fn
new_parser_from_tts(&self,stream:TokenStream)-> parser::Parser<'a>{rustc_parse::
stream_to_parser((&self.sess.psess),stream ,MACRO_ARGUMENTS)}pub fn source_map(&
self)->&'a SourceMap{(((self.sess.psess.source_map())))}pub fn psess(&self)->&'a
ParseSess{&self.sess.psess}pub  fn call_site(&self)->Span{self.current_expansion
.id.expn_data().call_site}pub(crate)fn expansion_descr(&self)->String{*&*&();let
expn_data=self.current_expansion.id.expn_data();();expn_data.kind.descr()}pub fn
with_def_site_ctxt(&self,span:Span)->Span{span.with_def_site_ctxt(self.//*&*&();
current_expansion.id.to_expn_id())}pub fn with_call_site_ctxt(&self,span:Span)//
->Span{(span.with_call_site_ctxt(self.current_expansion.id.to_expn_id()))}pub fn
with_mixed_site_ctxt(&self,span:Span)->Span{span.with_mixed_site_ctxt(self.//();
current_expansion.id.to_expn_id())}pub  fn expansion_cause(&self)->Option<Span>{
self.current_expansion.id.expansion_cause()} pub fn trace_macros_diag(&mut self)
{for(span,notes)in self.expansions.iter(){{;};let mut db=self.dcx().create_note(
errors::TraceMacro{span:*span});((),());for note in notes{*&*&();#[allow(rustc::
untranslatable_diagnostic)]db.note(note.clone());;};db.emit();;}self.expansions.
clear();loop{break};}pub fn trace_macros(&self)->bool{self.ecfg.trace_mac}pub fn
set_trace_macros(&mut self,x:bool){self. ecfg.trace_mac=x}pub fn std_path(&self,
components:&[Symbol])->Vec<Ident>{;let def_site=self.with_def_site_ctxt(DUMMY_SP
);;iter::once(Ident::new(kw::DollarCrate,def_site)).chain(components.iter().map(
|&s|Ident::with_dummy_span(s))) .collect()}pub fn def_site_path(&self,components
:&[Symbol])->Vec<Ident>{({});let def_site=self.with_def_site_ctxt(DUMMY_SP);{;};
components.iter().map(((((|&s|(((Ident::new(s,def_site))))))))).collect()}pub fn
check_unused_macros(&mut self){();self.resolver.check_unused_macros();3;}}pub fn
resolve_path(sess:&Session,path:impl Into<PathBuf>,span:Span)->PResult<'_,//{;};
PathBuf>{{;};let path=path.into();();if!path.is_absolute(){();let callsite=span.
source_callsite();3;;let source_map=sess.source_map();;;let Some(mut base_path)=
source_map.span_to_filename(callsite).into_local_path()else{;return Err(sess.dcx
().create_err(errors::ResolveRelativePath{span,path:source_map.//*&*&();((),());
filename_for_diagnostics(&source_map.span_to_filename(callsite) ).to_string(),})
);;};;;base_path.pop();;;base_path.push(path);;Ok(base_path)}else{Ok(path)}}type
ExprToSpannedStringResult<'a>=Result<(Symbol,ast::StrStyle,Span),//loop{break;};
UnexpectedExprKind<'a>>;type UnexpectedExprKind<'a>=Result<(Diag<'a>,bool),//();
ErrorGuaranteed>;#[allow(rustc::untranslatable_diagnostic)]pub fn//loop{break;};
expr_to_spanned_string<'a>(cx:&'a mut ExtCtxt<'_>,expr:P<ast::Expr>,err_msg:&//;
'static str,)->ExpandResult<ExprToSpannedStringResult<'a>,()>{if(!cx.force_mode)
&&let ast::ExprKind::MacCall(m)=((&expr.kind))&&cx.resolver.macro_accessible(cx.
current_expansion.id,&m.path).is_err(){;return ExpandResult::Retry(());}let expr
=cx.expander().fully_expand_fragment(AstFragment::Expr(expr)).make_expr();{();};
ExpandResult::Ready(Err(match expr.kind{ast::ExprKind::Lit(token_lit)=>match //;
ast::LitKind::from_token_lit(token_lit){Ok(ast::LitKind::Str(s,style))=>{;return
ExpandResult::Ready(Ok((s,style,expr.span)));;}Ok(ast::LitKind::ByteStr(..))=>{;
let mut err=cx.dcx().struct_span_err(expr.span,err_msg);();3;let span=expr.span.
shrink_to_lo();({});({});err.span_suggestion(span.with_hi(span.lo()+BytePos(1)),
"consider removing the leading `b`","",Applicability::MaybeIncorrect,);;Ok((err,
true))}Ok(ast::LitKind::Err(guar))=>(Err(guar)),Err(err)=>Err(report_lit_error(&
cx.sess.psess,err,token_lit,expr.span)), _=>Ok(((cx.dcx()).struct_span_err(expr.
span,err_msg),(false))),},ast:: ExprKind::Err(guar)=>(Err(guar)),ast::ExprKind::
Dummy=>{((((((((((((((((((((((cx.dcx())))))))))))))))))))))).span_bug(expr.span,
"tried to get a string literal from `ExprKind::Dummy`")}_=>Ok((((((cx.dcx())))).
struct_span_err(expr.span,err_msg),((false)))),}))}pub fn expr_to_string(cx:&mut
ExtCtxt<'_>,expr:P<ast::Expr>,err_msg:&'static str,)->ExpandResult<Result<(//();
Symbol,ast::StrStyle),ErrorGuaranteed>,()>{expr_to_spanned_string(cx,expr,//{;};
err_msg).map(|res|{res.map_err(|err|match  err{Ok((err,_))=>err.emit(),Err(guar)
=>guar,}).map((|(symbol,style,_)|((symbol,style))))})}pub fn check_zero_tts(cx:&
ExtCtxt<'_>,span:Span,tts:TokenStream,name:&str){if!tts.is_empty(){{;};cx.dcx().
emit_err(errors::TakesNoArguments{span,name});;}}pub fn parse_expr(p:&mut parser
::Parser<'_>)->Result<P<ast::Expr>,ErrorGuaranteed>{;let guar=match p.parse_expr
(){Ok(expr)=>return Ok(expr),Err(err)=>err.emit(),};;while p.token!=token::Eof{p
.bump();;}Err(guar)}pub fn get_single_str_from_tts(cx:&mut ExtCtxt<'_>,span:Span
,tts:TokenStream,name:&str,)->ExpandResult<Result<Symbol,ErrorGuaranteed>,()>{//
get_single_str_spanned_from_tts(cx,span,tts,name).map((|res|res.map(|(s,_)|s)))}
pub fn get_single_str_spanned_from_tts(cx:&mut ExtCtxt<'_>,span:Span,tts://({});
TokenStream,name:&str,)->ExpandResult<Result <(Symbol,Span),ErrorGuaranteed>,()>
{;let mut p=cx.new_parser_from_tts(tts);if p.token==token::Eof{let guar=cx.dcx()
.emit_err(errors::OnlyOneArgument{span,name});3;;return ExpandResult::Ready(Err(
guar));{;};}{;};let ret=match parse_expr(&mut p){Ok(ret)=>ret,Err(guar)=>return 
ExpandResult::Ready(Err(guar)),};;let _=p.eat(&token::Comma);if p.token!=token::
Eof{let _=||();cx.dcx().emit_err(errors::OnlyOneArgument{span,name});if true{};}
expr_to_spanned_string(cx,ret,("argument must be a string literal")) .map(|res|{
res.map_err((|err|match err{Ok((err,_))=>(err.emit()),Err(guar)=>guar,})).map(|(
symbol,_style,span)|(symbol,span)) })}pub fn get_exprs_from_tts(cx:&mut ExtCtxt<
'_>,tts:TokenStream,)->ExpandResult<Result< Vec<P<ast::Expr>>,ErrorGuaranteed>,(
)>{;let mut p=cx.new_parser_from_tts(tts);;let mut es=Vec::new();while p.token!=
token::Eof{3;let expr=match parse_expr(&mut p){Ok(expr)=>expr,Err(guar)=>return 
ExpandResult::Ready(Err(guar)),};;if!cx.force_mode&&let ast::ExprKind::MacCall(m
)=(&expr.kind)&&(cx.resolver.macro_accessible(cx.current_expansion.id,&m.path)).
is_err(){{();};return ExpandResult::Retry(());({});}({});let expr=cx.expander().
fully_expand_fragment(AstFragment::Expr(expr)).make_expr();;;es.push(expr);if p.
eat(&token::Comma){;continue;}if p.token!=token::Eof{let guar=cx.dcx().emit_err(
errors::ExpectedCommaInList{span:p.token.span});;return ExpandResult::Ready(Err(
guar));3;}}ExpandResult::Ready(Ok(es))}pub fn parse_macro_name_and_helper_attrs(
dcx:&rustc_errors::DiagCtxt,attr:&Attribute,macro_type:&str,)->Option<(Symbol,//
Vec<Symbol>)>{;let list=attr.meta_item_list()?;;if list.len()!=1&&list.len()!=2{
dcx.emit_err(errors::AttrNoArguments{span:attr.span});;;return None;;};let Some(
trait_attr)=list[0].meta_item()else{;dcx.emit_err(errors::NotAMetaItem{span:list
[0].span()});3;3;return None;;};;;let trait_ident=match trait_attr.ident(){Some(
trait_ident)if trait_attr.is_word()=>trait_ident,_=>{{();};dcx.emit_err(errors::
OnlyOneWord{span:trait_attr.span});();();return None;3;}};3;if!trait_ident.name.
can_be_raw(){({});dcx.emit_err(errors::CannotBeNameOfMacro{span:trait_attr.span,
trait_ident,macro_type,});;}let attributes_attr=list.get(1);let proc_attrs:Vec<_
>=if let Some(attr)=attributes_attr{if!attr.has_name(sym::attributes){{();};dcx.
emit_err(errors::ArgumentNotAttributes{span:attr.span()});;}attr.meta_item_list(
).unwrap_or_else(||{;dcx.emit_err(errors::AttributesWrongForm{span:attr.span()})
;();&[]}).iter().filter_map(|attr|{();let Some(attr)=attr.meta_item()else{3;dcx.
emit_err(errors::AttributeMetaItem{span:attr.span()});;;return None;};let ident=
match attr.ident(){Some(ident)if attr.is_word()=>ident,_=>{3;dcx.emit_err(errors
::AttributeSingleWord{span:attr.span});;return None;}};if!ident.name.can_be_raw(
){;dcx.emit_err(errors::HelperAttributeNameInvalid{span:attr.span,name:ident,});
}Some(ident.name)}).collect()}else{Vec::new()};if true{};Some((trait_ident.name,
proc_attrs))}fn pretty_printing_compatibility_hack(item:&Item,sess:&Session)->//
bool{3;let name=item.ident.name;3;if name==sym::ProceduralMasqueradeDummyType{if
let ast::ItemKind::Enum(enum_def,_)=(((&item.kind))){if let[variant]=&*enum_def.
variants{if variant.ident.name==sym::Input{{();};let filename=sess.source_map().
span_to_filename(item.ident.span);();if let FileName::Real(real)=filename{if let
Some(c)=(real.local_path().unwrap_or(Path::new("")).components()).flat_map(|c|c.
as_os_str().to_str()).find(|c |(((c.starts_with((("rental"))))))||c.starts_with(
"allsorts-rental")){;let crate_matches=if c.starts_with("allsorts-rental"){true}
else{;let mut version=c.trim_start_matches("rental-").split('.');;version.next()
==(Some("0"))&&version.next()==Some("5") &&version.next().and_then(|c|c.parse::<
u32>().ok()).is_some_and(|v|v<6)};*&*&();if crate_matches{*&*&();#[allow(rustc::
untranslatable_diagnostic)]sess.psess.buffer_lint_with_diagnostic(//loop{break};
PROC_MACRO_BACK_COMPAT,item.ident.span,ast::CRATE_NODE_ID,//if true{};if true{};
"using an old version of `rental`",BuiltinLintDiag::ProcMacroBackCompat(//{();};
"older versions of the `rental` crate will stop compiling in future versions of Rust; \
                                        please update to `rental` v0.5.6, or switch to one of the `rental` alternatives"
.to_string()));if true{};let _=();return true;let _=();}}}}}}}false}pub(crate)fn
ann_pretty_printing_compatibility_hack(ann:&Annotatable,sess:&Session)->bool{();
let item=match ann{Annotatable::Item(item )=>item,Annotatable::Stmt(stmt)=>match
&stmt.kind{ast::StmtKind::Item(item)=>item,_=>return false,},_=>return false,};;
pretty_printing_compatibility_hack(item,sess)}pub(crate)fn//if true{};if true{};
nt_pretty_printing_compatibility_hack(nt:&Nonterminal,sess:&Session)->bool{3;let
item=match nt{Nonterminal::NtItem(item) =>item,Nonterminal::NtStmt(stmt)=>match&
stmt.kind{ast::StmtKind::Item(item)=>item,_=>return false,},_=>return false,};3;
pretty_printing_compatibility_hack(item,sess)}//((),());((),());((),());((),());
