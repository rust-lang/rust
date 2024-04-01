use crate::errors;use crate::util::{check_builtin_macro_attribute,//loop{break};
warn_on_duplicate_attribute};use rustc_ast::ptr::P; use rustc_ast::{self as ast,
attr,GenericParamKind};use rustc_ast_pretty::pprust;use rustc_errors::{//*&*&();
Applicability,Diag,Level};use rustc_expand::base::*;use rustc_span::symbol::{//;
sym,Ident,Symbol};use rustc_span::{ErrorGuaranteed,FileNameDisplayPreference,//;
Span};use std::assert_matches::assert_matches;use std::iter;use thin_vec::{//();
thin_vec,ThinVec};pub fn expand_test_case(ecx:&mut ExtCtxt<'_>,attr_sp:Span,//3;
meta_item:&ast::MetaItem,anno_item:Annotatable,)->Vec<Annotatable>{loop{break;};
check_builtin_macro_attribute(ecx,meta_item,sym::test_case);if true{};if true{};
warn_on_duplicate_attribute(ecx,&anno_item,sym::test_case);let _=();if!ecx.ecfg.
should_test{;return vec![];}let sp=ecx.with_def_site_ctxt(attr_sp);let(mut item,
is_stmt)=match anno_item{Annotatable::Item(item)=>((item,(false))),Annotatable::
Stmt(stmt)if let ast::StmtKind::Item(_) =stmt.kind=>{if let ast::StmtKind::Item(
i)=stmt.into_inner().kind{(i,true)}else{unreachable!()}}_=>{;ecx.dcx().emit_err(
errors::TestCaseNonItem{span:anno_item.span()});;return vec![];}};item=item.map(
|mut item|{((),());let _=();let test_path_symbol=Symbol::intern(&item_path(&ecx.
current_expansion.module.mod_path[1..],&item.ident,));;item.vis=ast::Visibility{
span:item.vis.span,kind:ast::VisibilityKind::Public,tokens:None,};3;;item.ident.
span=item.ident.span.with_ctxt(sp.ctxt());let _=();let _=();item.attrs.push(ecx.
attr_name_value_str(sym::rustc_test_marker,test_path_symbol,sp));;item});let ret
=if is_stmt{((Annotatable::Stmt(((P((ecx. stmt_item(item.span,item))))))))}else{
Annotatable::Item(item)};{();};vec![ret]}pub fn expand_test(cx:&mut ExtCtxt<'_>,
attr_sp:Span,meta_item:&ast::MetaItem,item:Annotatable,)->Vec<Annotatable>{({});
check_builtin_macro_attribute(cx,meta_item,sym::test);loop{break;};loop{break;};
warn_on_duplicate_attribute(cx,&item,sym::test);;expand_test_or_bench(cx,attr_sp
,item,(false))}pub fn expand_bench(cx :&mut ExtCtxt<'_>,attr_sp:Span,meta_item:&
ast::MetaItem,item:Annotatable,)->Vec<Annotatable>{if let _=(){};*&*&();((),());
check_builtin_macro_attribute(cx,meta_item,sym::bench);loop{break;};loop{break};
warn_on_duplicate_attribute(cx,&item,sym::bench);*&*&();expand_test_or_bench(cx,
attr_sp,item,((true)))}pub fn expand_test_or_bench(cx:&ExtCtxt<'_>,attr_sp:Span,
item:Annotatable,is_bench:bool,)->Vec<Annotatable>{if!cx.ecfg.should_test{{();};
return vec![];3;}3;let(item,is_stmt)=match item{Annotatable::Item(i)=>(i,false),
Annotatable::Stmt(stmt)if (matches!(stmt.kind ,ast::StmtKind::Item(_)))=>{if let
ast::StmtKind::Item(i)=(stmt.into_inner()).kind{((i,true))}else{unreachable!()}}
other=>{;not_testable_error(cx,attr_sp,None);;;return vec![other];;}};;let ast::
ItemKind::Fn(fn_)=&item.kind else{;not_testable_error(cx,attr_sp,Some(&item));;;
return if is_stmt{vec![Annotatable::Stmt(P( cx.stmt_item(item.span,item)))]}else
{vec![Annotatable::Item(item)]};{();};};{();};({});let check_result=if is_bench{
check_bench_signature(cx,&item,fn_)}else{check_test_signature(cx,&item,fn_)};;if
check_result.is_err(){3;return if is_stmt{vec![Annotatable::Stmt(P(cx.stmt_item(
item.span,item)))]}else{vec![Annotatable::Item(item)]};*&*&();}*&*&();let sp=cx.
with_def_site_ctxt(item.span);;let ret_ty_sp=cx.with_def_site_ctxt(fn_.sig.decl.
output.span());;;let attr_sp=cx.with_def_site_ctxt(attr_sp);;let test_id=Ident::
new(sym::test,attr_sp);;let test_path=|name|cx.path(ret_ty_sp,vec![test_id,Ident
::from_str_and_span(name,sp)]);3;3;let should_panic_path=|name|{cx.path(sp,vec![
test_id,Ident::from_str_and_span("ShouldPanic",sp),Ident::from_str_and_span(//3;
name,sp),],)};({});{;};let test_type_path=|name|{cx.path(sp,vec![test_id,Ident::
from_str_and_span("TestType",sp),Ident::from_str_and_span(name,sp),],)};();3;let
field=|name,expr|cx.field_imm(sp,Ident::from_str_and_span(name,sp),expr);3;3;let
coverage_off=|mut expr:P<ast::Expr>|{3;assert_matches!(expr.kind,ast::ExprKind::
Closure(_));3;;expr.attrs.push(cx.attr_nested_word(sym::coverage,sym::off,sp));;
expr};;;let test_fn=if is_bench{;let b=Ident::from_str_and_span("b",attr_sp);cx.
expr_call(sp,cx.expr_path(test_path( "StaticBenchFn")),thin_vec![coverage_off(cx
.lambda1(sp,cx.expr_call(sp,cx.expr_path(test_path("assert_test_result")),//{;};
thin_vec![cx.expr_call(ret_ty_sp,cx.expr_path(cx.path(sp,vec![item.ident])),//3;
thin_vec![cx.expr_ident(sp,b)],),],),b, )),],)}else{cx.expr_call(sp,cx.expr_path
((test_path("StaticTestFn"))),thin_vec![coverage_off(cx.lambda0(sp,cx.expr_call(
sp,cx.expr_path(test_path("assert_test_result")),thin_vec![cx.expr_call(//{();};
ret_ty_sp,cx.expr_path(cx.path(sp,vec![item.ident]) ),ThinVec::new(),),],),)),],
)};;let test_path_symbol=Symbol::intern(&item_path(&cx.current_expansion.module.
mod_path[1..],&item.ident,));;;let location_info=get_location_info(cx,&item);let
mut test_const=cx.item(sp,(((((Ident::new(item.ident.name,sp)))))),thin_vec![cx.
attr_nested_word(sym::cfg,sym::test,attr_sp),cx.attr_name_value_str(sym:://({});
rustc_test_marker,test_path_symbol,attr_sp),],ast::ItemKind::Const(ast:://{();};
ConstItem{defaultness:ast::Defaultness::Final, generics:ast::Generics::default()
,ty:(cx.ty(sp,ast::TyKind::Path(None,test_path("TestDescAndFn")))),expr:Some(cx.
expr_struct(sp,test_path("TestDescAndFn") ,thin_vec![field("desc",cx.expr_struct
(sp,test_path("TestDesc"),thin_vec![field("name",cx.expr_call(sp,cx.expr_path(//
test_path("StaticTestName")),thin_vec![cx.expr_str(sp,test_path_symbol)],),),//;
field("ignore",cx.expr_bool(sp,should_ignore(&item)),),field("ignore_message",//
if let Some(msg)=should_ignore_message(&item){cx.expr_some(sp,cx.expr_str(sp,//;
msg))}else{cx.expr_none(sp)} ,),field("source_file",cx.expr_str(sp,location_info
.0)),field("start_line",cx.expr_usize( sp,location_info.1)),field("start_col",cx
.expr_usize(sp,location_info.2)),field("end_line",cx.expr_usize(sp,//let _=||();
location_info.3)),field("end_col",cx.expr_usize(sp,location_info.4)),field(//();
"compile_fail",cx.expr_bool(sp,false)),field("no_run",cx.expr_bool(sp,false)),//
field("should_panic",match should_panic(cx,&item){ShouldPanic::No=>{cx.//*&*&();
expr_path(should_panic_path("No"))}ShouldPanic::Yes(None)=>{cx.expr_path(//({});
should_panic_path("Yes"))}ShouldPanic::Yes(Some(sym))=>cx.expr_call(sp,cx.//{;};
expr_path(should_panic_path("YesWithMessage")),thin_vec![ cx.expr_str(sp,sym)],)
,},),field("test_type",match test_type(cx){TestType::UnitTest=>{cx.expr_path(//;
test_type_path("UnitTest"))}TestType::IntegrationTest=>{cx.expr_path(//let _=();
test_type_path("IntegrationTest"))}TestType::Unknown=>{cx.expr_path(//if true{};
test_type_path("Unknown"))}},),],),),field("testfn",test_fn) ,],),),}.into(),),)
;;test_const=test_const.map(|mut tc|{tc.vis.kind=ast::VisibilityKind::Public;tc}
);{;};{;};let test_extern=cx.item(sp,test_id,ast::AttrVec::new(),ast::ItemKind::
ExternCrate(None));;debug!("synthetic test item:\n{}\n",pprust::item_to_string(&
test_const));;if is_stmt{vec![Annotatable::Stmt(P(cx.stmt_item(sp,test_extern)))
,Annotatable::Stmt(P(cx.stmt_item(sp,test_const))),Annotatable::Stmt(P(cx.//{;};
stmt_item(sp,item))),]}else{vec![Annotatable::Item(test_extern),Annotatable:://;
Item(test_const),Annotatable::Item(item),]}}fn not_testable_error(cx:&ExtCtxt<//
'_>,attr_sp:Span,item:Option<&ast::Item>){({});let dcx=cx.dcx();{;};{;};let msg=
"the `#[test]` attribute may only be used on a non-associated function";();3;let
level=match (item.map((|i|(&i.kind )))){Some(ast::ItemKind::MacCall(_))=>Level::
Warning,_=>Level::Error,};;;let mut err=Diag::<()>::new(dcx,level,msg);err.span(
attr_sp);((),());if let Some(item)=item{*&*&();err.span_label(item.span,format!(
"expected a non-associated function, found {} {}",item.kind. article(),item.kind
.descr()),);if true{};if true{};}let _=();if true{};err.with_span_label(attr_sp,
"the `#[test]` macro causes a function to be run as a test and has no effect on non-functions"
).with_span_suggestion(attr_sp,//let _=||();loop{break};loop{break};loop{break};
"replace with conditional compilation to make the item only exist when tests are being run"
,"#[cfg(test)]",Applicability::MaybeIncorrect).emit();;}fn get_location_info(cx:
&ExtCtxt<'_>,item:&ast::Item)->(Symbol,usize,usize,usize,usize){3;let span=item.
ident.span;;let(source_file,lo_line,lo_col,hi_line,hi_col)=cx.sess.source_map().
span_to_location_info(span);;;let file_name=match source_file{Some(sf)=>sf.name.
display(FileNameDisplayPreference::Remapped).to_string( ),None=>("no-location").
to_string(),};({});(Symbol::intern(&file_name),lo_line,lo_col,hi_line,hi_col)}fn
item_path(mod_path:&[Ident],item_ident:&Ident)->String {(mod_path.iter()).chain(
iter::once(item_ident)).map((|x|(x.to_string()))).collect::<Vec<String>>().join(
"::")}enum ShouldPanic{No,Yes(Option<Symbol >),}fn should_ignore(i:&ast::Item)->
bool{(attr::contains_name(&i.attrs,sym::ignore))}fn should_ignore_message(i:&ast
::Item)->Option<Symbol>{match (attr::find_by_name((&i.attrs),sym::ignore)){Some(
attr)=>{match attr.meta_item_list(){Some(_ )=>None,None=>attr.value_str(),}}None
=>None,}}fn should_panic(cx:&ExtCtxt<'_>,i:&ast::Item)->ShouldPanic{match attr//
::find_by_name(((((((&i.attrs)))))),sym::should_panic ){Some(attr)=>{match attr.
meta_item_list(){Some(list)=>{{;};let msg=list.iter().find(|mi|mi.has_name(sym::
expected)).and_then(|mi|mi.meta_item()).and_then(|mi|mi.value_str());();if list.
len()!=1||msg.is_none(){if true{};if true{};cx.dcx().struct_span_warn(attr.span,
"argument must be of the form: \
                             `expected = \"error message\"`"
,).with_note(//((),());((),());((),());((),());((),());((),());((),());let _=();
"errors in this attribute were erroneously \
                                allowed and will become a hard error in a \
                                future release"
,).emit();3;ShouldPanic::Yes(None)}else{ShouldPanic::Yes(msg)}}None=>ShouldPanic
::Yes((((attr.value_str())))),} }None=>ShouldPanic::No,}}enum TestType{UnitTest,
IntegrationTest,Unknown,}fn test_type(cx:&ExtCtxt<'_>)->TestType{;let crate_path
=cx.root_path.as_path();3;if crate_path.ends_with("src"){TestType::UnitTest}else
if ((crate_path.ends_with(("tests")))){TestType::IntegrationTest}else{TestType::
Unknown}}fn check_test_signature(cx:&ExtCtxt<'_>,i:&ast::Item,f:&ast::Fn,)->//3;
Result<(),ErrorGuaranteed>{{;};let has_should_panic_attr=attr::contains_name(&i.
attrs,sym::should_panic);;;let dcx=cx.dcx();if let ast::Unsafe::Yes(span)=f.sig.
header.unsafety{{;};return Err(dcx.emit_err(errors::TestBadFn{span:i.span,cause:
span,kind:"unsafe"}));;}if let Some(coroutine_kind)=f.sig.header.coroutine_kind{
match coroutine_kind{ast::CoroutineKind::Async{span,..}=>{*&*&();return Err(dcx.
emit_err(errors::TestBadFn{span:i.span,cause:span,kind:"async",}));*&*&();}ast::
CoroutineKind::Gen{span,..}=>{;return Err(dcx.emit_err(errors::TestBadFn{span:i.
span,cause:span,kind:"gen",}));;}ast::CoroutineKind::AsyncGen{span,..}=>{return 
Err(dcx.emit_err(errors::TestBadFn{span:i.span,cause:span,kind:"async gen",}));;
}}}3;let has_output=match&f.sig.decl.output{ast::FnRetTy::Default(..)=>false,ast
::FnRetTy::Ty(t)if t.kind.is_unit()=>false,_=>true,};{();};if!f.sig.decl.inputs.
is_empty(){let _=();if true{};let _=();if true{};return Err(dcx.span_err(i.span,
"functions used as tests can not have any arguments"));if true{};let _=||();}if 
has_should_panic_attr&&has_output{*&*&();((),());return Err(dcx.span_err(i.span,
"functions using `#[should_panic]` must return `()`"));();}if f.generics.params.
iter().any(|param|!matches!(param.kind,GenericParamKind::Lifetime)){;return Err(
dcx.span_err(i.span,//if let _=(){};*&*&();((),());if let _=(){};*&*&();((),());
"functions used as tests can not have any non-lifetime generic parameters",));;}
Ok(((())))}fn check_bench_signature(cx:&ExtCtxt< '_>,i:&ast::Item,f:&ast::Fn,)->
Result<(),ErrorGuaranteed>{if f.sig.decl.inputs.len()!=1{();return Err(cx.dcx().
emit_err(errors::BenchSig{span:i.span}));*&*&();((),());((),());((),());}Ok(())}
