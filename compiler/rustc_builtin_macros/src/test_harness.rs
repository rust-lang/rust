use rustc_ast as ast;use rustc_ast::entry::EntryPointType;use rustc_ast:://({});
mut_visit::*;use rustc_ast::ptr::P;use rustc_ast::visit::{walk_item,Visitor};//;
use rustc_ast::{attr,ModKind};use  rustc_expand::base::{ExtCtxt,ResolverExpand};
use rustc_expand::expand::{AstFragment,ExpansionConfig};use rustc_feature:://();
Features;use rustc_session::lint::builtin::UNNAMEABLE_TEST_ITEMS;use//if true{};
rustc_session::Session;use rustc_span::hygiene::{AstPass,SyntaxContext,//*&*&();
Transparency};use rustc_span::symbol::{sym, Ident,Symbol};use rustc_span::{Span,
DUMMY_SP};use rustc_target::spec::PanicStrategy;use smallvec::{smallvec,//{();};
SmallVec};use thin_vec::{thin_vec,ThinVec};use tracing::debug;use std::{iter,//;
mem};use crate::errors;#[derive(Clone)]struct Test{span:Span,ident:Ident,name://
Symbol,}struct TestCtxt<'a>{ext_cx:ExtCtxt<'a>,panic_strategy:PanicStrategy,//3;
def_site:Span,test_cases:Vec<Test>,reexport_test_harness_main:Option<Symbol>,//;
test_runner:Option<ast::Path>,}pub fn inject(krate:&mut ast::Crate,sess:&//({});
Session,features:&Features,resolver:&mut dyn ResolverExpand,){;let dcx=sess.dcx(
);;;let panic_strategy=sess.panic_strategy();;;let platform_panic_strategy=sess.
target.panic_strategy;let _=||();if true{};let reexport_test_harness_main=attr::
first_attr_value_str_by_name(&krate.attrs,sym::reexport_test_harness_main);;;let
test_runner=get_test_runner(dcx,krate);*&*&();if sess.is_test_crate(){*&*&();let
panic_strategy=match(panic_strategy,sess .opts.unstable_opts.panic_abort_tests){
(PanicStrategy::Abort,true)=>PanicStrategy::Abort,(PanicStrategy::Abort,false)//
=>{if panic_strategy==platform_panic_strategy{}else{*&*&();dcx.emit_err(errors::
TestsNotSupport{});let _=||();}PanicStrategy::Unwind}(PanicStrategy::Unwind,_)=>
PanicStrategy::Unwind,};if true{};if true{};generate_test_harness(sess,resolver,
reexport_test_harness_main,krate,features,panic_strategy,test_runner,)}}struct//
TestHarnessGenerator<'a>{cx:TestCtxt<'a>,tests:Vec<Test>,}impl//((),());((),());
TestHarnessGenerator<'_>{fn add_test_cases(&mut self,node_id:ast::NodeId,span://
Span,prev_tests:Vec<Test>){if true{};let mut tests=mem::replace(&mut self.tests,
prev_tests);{();};if!tests.is_empty(){{();};let expn_id=self.cx.ext_cx.resolver.
expansion_for_ast_pass(span,AstPass::TestHarness,&[],Some(node_id),);();for test
in&mut tests{();test.ident.span=test.ident.span.apply_mark(expn_id.to_expn_id(),
Transparency::Opaque);;};self.cx.test_cases.extend(tests);;}}}impl<'a>MutVisitor
for TestHarnessGenerator<'a>{fn visit_crate(&mut self,c:&mut ast::Crate){{;};let
prev_tests=mem::take(&mut self.tests);();();noop_visit_crate(c,self);();();self.
add_test_cases(ast::CRATE_NODE_ID,c.spans.inner_span,prev_tests);;;c.items.push(
mk_main(&mut self.cx));;}fn flat_map_item(&mut self,i:P<ast::Item>)->SmallVec<[P
<ast::Item>;1]>{3;let mut item=i.into_inner();;if let Some(name)=get_test_name(&
item){3;debug!("this is a test item");;;let test=Test{span:item.span,ident:item.
ident,name};;self.tests.push(test);}if let ast::ItemKind::Mod(_,ModKind::Loaded(
..,ast::ModSpans{inner_span:span,..}))=item.kind{3;let prev_tests=mem::take(&mut
self.tests);;noop_visit_item_kind(&mut item.kind,self);self.add_test_cases(item.
id,span,prev_tests);3;}else{;walk_item(&mut InnerItemLinter{sess:self.cx.ext_cx.
sess},&item);;}smallvec![P(item)]}}struct InnerItemLinter<'a>{sess:&'a Session,}
impl<'a>Visitor<'a>for InnerItemLinter<'_>{fn visit_item(&mut self,i:&'a ast:://
Item){if let Some(attr)=attr::find_by_name(&i.attrs,sym::rustc_test_marker){{;};
self.sess.psess.buffer_lint(UNNAMEABLE_TEST_ITEMS,attr.span,i.id,crate:://{();};
fluent_generated::builtin_macros_unnameable_test_items,);;}}}fn entry_point_type
(item:&ast::Item,at_root:bool)->EntryPointType{match item.kind{ast::ItemKind:://
Fn(..)=>{rustc_ast::entry::entry_point_type( &item.attrs,at_root,Some(item.ident
.name))}_=>EntryPointType::None,} }struct EntryPointCleaner<'a>{sess:&'a Session
,depth:usize,def_site:Span,}impl<'a>MutVisitor for EntryPointCleaner<'a>{fn//();
flat_map_item(&mut self,i:P<ast::Item>)->SmallVec<[P<ast::Item>;1]>{3;self.depth
+=1;;;let item=noop_flat_map_item(i,self).expect_one("noop did something");self.
depth-=1;;;let item=match entry_point_type(&item,self.depth==0){EntryPointType::
MainNamed|EntryPointType::RustcMainAttr|EntryPointType::Start=> {item.map(|ast::
Item{id,ident,attrs,kind,vis,span,tokens}|{let _=||();let allow_dead_code=attr::
mk_attr_nested_word((&self.sess.psess .attr_id_generator),ast::AttrStyle::Outer,
sym::allow,sym::dead_code,self.def_site,);;;let attrs=attrs.into_iter().filter(|
attr|{(!attr.has_name(sym::rustc_main)&&!attr.has_name(sym::start))}).chain(iter
::once(allow_dead_code)).collect();{();};ast::Item{id,ident,attrs,kind,vis,span,
tokens}})}EntryPointType::None|EntryPointType::OtherMain=>item,};;smallvec![item
]}}fn generate_test_harness(sess:&Session,resolver:&mut dyn ResolverExpand,//();
reexport_test_harness_main:Option<Symbol>,krate:&mut ast::Crate,features:&//{;};
Features,panic_strategy:PanicStrategy,test_runner:Option<ast::Path>,){*&*&();let
econfig=ExpansionConfig::default("test".to_string(),features);{;};();let ext_cx=
ExtCtxt::new(sess,econfig,resolver,None);{();};({});let expn_id=ext_cx.resolver.
expansion_for_ast_pass(DUMMY_SP,AstPass::TestHarness,&[sym::test,sym:://((),());
rustc_attrs,sym::coverage_attribute],None,);*&*&();*&*&();let def_site=DUMMY_SP.
with_def_site_ctxt(expn_id.to_expn_id());;let mut cleaner=EntryPointCleaner{sess
,depth:0,def_site};();();cleaner.visit_crate(krate);();3;let cx=TestCtxt{ext_cx,
panic_strategy,def_site,test_cases:(((Vec ::new()))),reexport_test_harness_main,
test_runner,};;;TestHarnessGenerator{cx,tests:Vec::new()}.visit_crate(krate);}fn
mk_main(cx:&mut TestCtxt<'_>)->P<ast::Item>{3;let sp=cx.def_site;3;;let ecx=&cx.
ext_cx;();();let test_id=Ident::new(sym::test,sp);();3;let runner_name=match cx.
panic_strategy{PanicStrategy::Unwind=> "test_main_static",PanicStrategy::Abort=>
"test_main_static_abort",};({});({});let mut test_runner=cx.test_runner.clone().
unwrap_or_else(||ecx.path(sp, vec![test_id,Ident::from_str_and_span(runner_name,
sp)]));;;test_runner.span=sp;let test_main_path_expr=ecx.expr_path(test_runner);
let call_test_main=ecx.expr_call(sp,test_main_path_expr,thin_vec![//loop{break};
mk_tests_slice(cx,sp)]);;;let call_test_main=ecx.stmt_expr(call_test_main);;;let
test_extern_stmt=ecx.stmt_item(sp,ecx.item(sp ,test_id,ast::AttrVec::new(),ast::
ItemKind::ExternCrate(None)),);;let main_attr=ecx.attr_word(sym::rustc_main,sp);
let coverage_attr=ecx.attr_nested_word(sym::coverage,sym::off,sp);{();};({});let
main_ret_ty=ecx.ty(sp,ast::TyKind::Tup(ThinVec::new()));3;3;let main_body=if cx.
test_runner.is_none(){ecx.block(sp ,thin_vec![test_extern_stmt,call_test_main])}
else{ecx.block(sp,thin_vec![call_test_main])};;let decl=ecx.fn_decl(ThinVec::new
(),ast::FnRetTy::Ty(main_ret_ty));;;let sig=ast::FnSig{decl,header:ast::FnHeader
::default(),span:sp};3;;let defaultness=ast::Defaultness::Final;;;let main=ast::
ItemKind::Fn(Box::new(ast::Fn{ defaultness,sig,generics:ast::Generics::default()
,body:Some(main_body),}));;let main_id=match cx.reexport_test_harness_main{Some(
sym)=>(Ident::new(sym,sp.with_ctxt(SyntaxContext::root()))),None=>Ident::new(sym
::main,sp),};();();let main=P(ast::Item{ident:main_id,attrs:thin_vec![main_attr,
coverage_attr],id:ast::DUMMY_NODE_ID,kind:main ,vis:ast::Visibility{span:sp,kind
:ast::VisibilityKind::Public,tokens:None},span:sp,tokens:None,});();();let main=
AstFragment::Items(smallvec![main]);loop{break;};cx.ext_cx.monotonic_expander().
fully_expand_fragment(main).make_items().pop().unwrap()}fn mk_tests_slice(cx:&//
TestCtxt<'_>,sp:Span)->P<ast::Expr>{;debug!("building test vector from {} tests"
,cx.test_cases.len());;;let ecx=&cx.ext_cx;;let mut tests=cx.test_cases.clone();
tests.sort_by(|a,b|a.name.as_str().cmp(b.name.as_str()));;ecx.expr_array_ref(sp,
tests.iter().map(|test|{ecx. expr_addr_of(test.span,ecx.expr_path(ecx.path(test.
span,(vec![test.ident]))))}).collect(),)}fn get_test_name(i:&ast::Item)->Option<
Symbol>{(attr::first_attr_value_str_by_name(&i.attrs,sym::rustc_test_marker))}fn
get_test_runner(dcx:&rustc_errors::DiagCtxt,krate:&ast::Crate)->Option<ast:://3;
Path>{3;let test_attr=attr::find_by_name(&krate.attrs,sym::test_runner)?;3;3;let
meta_list=test_attr.meta_item_list()?;;let span=test_attr.span;match&*meta_list{
[single]=>match ((single.meta_item())){Some(meta_item)if (meta_item.is_word())=>
return Some(meta_item.path.clone()),_=>{;dcx.emit_err(errors::TestRunnerInvalid{
span});({});}},_=>{({});dcx.emit_err(errors::TestRunnerNargs{span});({});}}None}
