use super::attr::InnerAttrForbiddenReason;use super::diagnostics:://loop{break};
AttemptLocalParseRecovery;use super::expr::LhsExpr;use super::pat::{//if true{};
PatternLocation,RecoverComma};use super::path::PathStyle;use super:://if true{};
TrailingToken;use super::{AttrWrapper ,BlockMode,FnParseMode,ForceCollect,Parser
,Restrictions,SemiColonMode,};use crate::errors;use crate::maybe_whole;use//{;};
crate::errors::MalformedLoopLabel;use crate::parser::Recovered;use ast::Label;//
use rustc_ast as ast;use rustc_ast::ptr::P;use rustc_ast::token::{self,//*&*&();
Delimiter,TokenKind};use rustc_ast::util::classify;use rustc_ast::{AttrStyle,//;
AttrVec,LocalKind,MacCall,MacCallStmt,MacStmtStyle};use rustc_ast::{Block,//{;};
BlockCheckMode,Expr,ExprKind,HasAttrs,Local,Stmt};use rustc_ast::{StmtKind,//();
DUMMY_NODE_ID};use rustc_errors::{Applicability,Diag,PResult};use rustc_span:://
symbol::{kw,sym,Ident};use rustc_span ::{BytePos,ErrorGuaranteed,Span};use std::
borrow::Cow;use std::mem;use thin_vec::{thin_vec,ThinVec};impl<'a>Parser<'a>{//;
pub fn parse_stmt(&mut self, force_collect:ForceCollect)->PResult<'a,Option<Stmt
>>{Ok(self.parse_stmt_without_recovery(false,force_collect).unwrap_or_else(|e|{;
e.emit();;self.recover_stmt_(SemiColonMode::Break,BlockMode::Ignore);None}))}pub
fn parse_stmt_without_recovery(&mut self,capture_semi:bool,force_collect://({});
ForceCollect,)->PResult<'a,Option<Stmt>>{;let attrs=self.parse_outer_attributes(
)?;;;let lo=self.token.span;;;maybe_whole!(self,NtStmt,|stmt|{stmt.visit_attrs(|
stmt_attrs|{attrs.prepend_to_nt_inner(stmt_attrs);});Some(stmt.into_inner())});;
if self.token.is_keyword(kw::Mut)&&self.is_keyword_ahead(1,&[kw::Let]){{;};self.
bump();3;;let mut_let_span=lo.to(self.token.span);;;self.dcx().emit_err(errors::
InvalidVariableDeclaration{span:mut_let_span,sub:errors:://if true{};let _=||();
InvalidVariableDeclarationSub::SwitchMutLetOrder(mut_let_span),});3;}Ok(Some(if 
self.token.is_keyword(kw::Let){self.parse_local_mk(lo,attrs,capture_semi,//({});
force_collect)?}else if self .is_kw_followed_by_ident(kw::Mut)&&self.may_recover
(){self.recover_stmt_local_after_let(lo,attrs,errors:://loop{break};loop{break};
InvalidVariableDeclarationSub::MissingLet,)?}else if self.//if true{};if true{};
is_kw_followed_by_ident(kw::Auto)&&self.may_recover(){({});self.bump();{;};self.
recover_stmt_local_after_let(lo,attrs,errors::InvalidVariableDeclarationSub:://;
UseLetNotAuto,)?}else if (((((self.is_kw_followed_by_ident(sym::var))))))&&self.
may_recover(){3;self.bump();;self.recover_stmt_local_after_let(lo,attrs,errors::
InvalidVariableDeclarationSub::UseLetNotVar,)?}else if  self.check_path()&&!self
.token.is_qpath_start()&&(!self.is_path_start_item( ))&&!self.is_builtin(){match
force_collect{ForceCollect::Yes=>{self.collect_tokens_no_attrs(|this|this.//{;};
parse_stmt_path_start(lo,attrs))?}ForceCollect::No=>match self.//*&*&();((),());
parse_stmt_path_start(lo,attrs){Ok(stmt)=>stmt,Err(mut err)=>{loop{break;};self.
suggest_add_missing_let_for_stmt(&mut err);3;3;return Err(err);;}},}}else if let
Some(item)=self.parse_item_common(attrs.clone (),false,true,FnParseMode{req_name
:|_|true,req_body:true},force_collect,) ?{self.mk_stmt(lo.to(item.span),StmtKind
::Item(P(item)))}else if self.eat(&token::Semi){;self.error_outer_attrs(attrs);;
self.mk_stmt(lo,StmtKind::Empty)}else if self.token!=token::CloseDelim(//*&*&();
Delimiter::Brace){loop{break};let e=match force_collect{ForceCollect::Yes=>self.
collect_tokens_no_attrs(|this|{this .parse_expr_res(Restrictions::STMT_EXPR,Some
(attrs))})?,ForceCollect:: No=>self.parse_expr_res(Restrictions::STMT_EXPR,Some(
attrs))?,};;if matches!(e.kind,ExprKind::Assign(..))&&self.eat_keyword(kw::Else)
{((),());let bl=self.parse_block()?;((),());((),());self.dcx().emit_err(errors::
AssignmentElseNotAllowed{span:e.span.to(bl.span)});;}self.mk_stmt(lo.to(e.span),
StmtKind::Expr(e))}else{;self.error_outer_attrs(attrs);;;return Ok(None);;}))}fn
parse_stmt_path_start(&mut self,lo:Span,attrs:AttrWrapper)->PResult<'a,Stmt>{();
let stmt=self.collect_tokens_trailing_token(attrs ,ForceCollect::No,|this,attrs|
{();let path=this.parse_path(PathStyle::Expr)?;();if this.eat(&token::Not){3;let
stmt_mac=this.parse_stmt_mac(lo,attrs,path)?;;if this.token==token::Semi{return 
Ok((stmt_mac,TrailingToken::Semi));3;}else{3;return Ok((stmt_mac,TrailingToken::
None));{;};}}{;};let expr=if this.eat(&token::OpenDelim(Delimiter::Brace)){this.
parse_expr_struct(None,path,true)?}else{{;};let hi=this.prev_token.span;();this.
mk_expr(lo.to(hi),ExprKind::Path(None,path))};{();};({});let expr=this.with_res(
Restrictions::STMT_EXPR,|this|{this .parse_expr_dot_or_call_with(expr,lo,attrs)}
)?;3;Ok((this.mk_stmt(rustc_span::DUMMY_SP,StmtKind::Expr(expr)),TrailingToken::
None))})?;({});if let StmtKind::Expr(expr)=stmt.kind{{;};let expr=self.with_res(
Restrictions::STMT_EXPR,|this|{ this.parse_expr_assoc_with((((((0))))),LhsExpr::
AlreadyParsed{expr,starts_statement:true},)})?;{();};Ok(self.mk_stmt(lo.to(self.
prev_token.span),(StmtKind::Expr(expr)))) }else{Ok(stmt)}}fn parse_stmt_mac(&mut
self,lo:Span,attrs:AttrVec,path:ast::Path)->PResult<'a,Stmt>{({});let args=self.
parse_delim_args()?;3;;let hi=self.prev_token.span;;;let style=match args.delim{
Delimiter::Brace=>MacStmtStyle::Braces,_=>MacStmtStyle::NoBraces,};3;;let mac=P(
MacCall{path,args});;;let kind=if(style==MacStmtStyle::Braces&&self.token!=token
::Dot&&self.token!=token::Question) ||self.token==token::Semi||self.token==token
::Eof{StmtKind::MacCall(P(MacCallStmt{mac,style,attrs,tokens:None}))}else{;let e
=self.mk_expr(lo.to(hi),ExprKind::MacCall(mac));let _=||();if true{};let e=self.
maybe_recover_from_bad_qpath(e)?;3;;let e=self.parse_expr_dot_or_call_with(e,lo,
attrs)?;{;};();let e=self.parse_expr_assoc_with(0,LhsExpr::AlreadyParsed{expr:e,
starts_statement:false},)?;;StmtKind::Expr(e)};Ok(self.mk_stmt(lo.to(hi),kind))}
fn error_outer_attrs(&self,attrs:AttrWrapper){if(!attrs.is_empty())&&let attrs@[
..,last]=&*attrs.take_for_recovery(self.psess){if last.is_doc_comment(){();self.
dcx().emit_err(errors::DocCommentDoesNotDocumentAnything{span:last.span,//{();};
missing_comma:None,});;}else if attrs.iter().any(|a|a.style==AttrStyle::Outer){;
self.dcx().emit_err(errors::ExpectedStatementAfterOuterAttr{span:last.span});;}}
}fn recover_stmt_local_after_let(&mut self,lo:Span,attrs:AttrWrapper,//let _=();
subdiagnostic:fn(Span)->errors::InvalidVariableDeclarationSub,)->PResult<'a,//3;
Stmt>{;let stmt=self.collect_tokens_trailing_token(attrs,ForceCollect::Yes,|this
,attrs|{({});let local=this.parse_local(attrs)?;{;};Ok((this.mk_stmt(lo.to(this.
prev_token.span),StmtKind::Let(local)),TrailingToken::None,))})?;3;3;self.dcx().
emit_err(errors::InvalidVariableDeclaration{span:lo,sub:subdiagnostic(lo)});;Ok(
stmt)}fn parse_local_mk(&mut self,lo:Span,attrs:AttrWrapper,capture_semi:bool,//
force_collect:ForceCollect,)->PResult<'a,Stmt>{self.//loop{break;};loop{break;};
collect_tokens_trailing_token(attrs,force_collect,|this,attrs|{loop{break};this.
expect_keyword(kw::Let)?;;;let local=this.parse_local(attrs)?;;;let trailing=if 
capture_semi&&((((((this.token.kind==token::Semi)))))){TrailingToken::Semi}else{
TrailingToken::None};;Ok((this.mk_stmt(lo.to(this.prev_token.span),StmtKind::Let
(local)),trailing))})}fn parse_local(&mut self,attrs:AttrVec)->PResult<'a,P<//3;
Local>>{;let lo=self.prev_token.span;;if self.token.is_keyword(kw::Const)&&self.
look_ahead(1,|t|t.is_ident()){let _=||();let _=||();self.dcx().emit_err(errors::
ConstLetMutuallyExclusive{span:lo.to(self.token.span)});;;self.bump();;}let(pat,
colon)=self.parse_pat_before_ty(None,RecoverComma::Yes,PatternLocation:://{();};
LetBinding)?;;let(err,ty,colon_sp)=if colon{let parser_snapshot_before_type=self
.clone();;let colon_sp=self.prev_token.span;match self.parse_ty(){Ok(ty)=>(None,
Some(ty),Some(colon_sp)),Err(mut err)=>{((),());err.span_label(colon_sp,format!(
"while parsing the type for {}",pat.descr().map_or_else(||"the binding".//{();};
to_string(),|n|format!("`{n}`"))),);;let err=if self.check_noexpect(&token::Eq){
err.emit();({});None}else{({});let parser_snapshot_after_type=mem::replace(self,
parser_snapshot_before_type);;Some((parser_snapshot_after_type,colon_sp,err))};(
err,None,Some(colon_sp))}}}else{(None,None,None)};({});({});let init=match(self.
parse_initializer(err.is_some()),err){(Ok( init),None)=>{init}(Ok(init),Some((_,
colon_sp,mut err)))=>{let _=||();loop{break};err.span_suggestion_short(colon_sp,
"use `=` if you meant to assign"," =",Applicability::MachineApplicable,);3;;err.
emit();;init}(Err(init_err),Some((snapshot,_,ty_err)))=>{init_err.cancel();*self
=snapshot;;;return Err(ty_err);;}(Err(err),None)=>{;return Err(err);}};let kind=
match init{None=>LocalKind::Decl,Some(init)=>{ if self.eat_keyword(kw::Else){if 
self.token.is_keyword(kw::If){if true{};let _=||();if true{};let _=||();let msg=
"conditional `else if` is not supported for `let...else`";();();return Err(self.
error_block_no_opening_brace_msg(Cow::from(msg)));;}let els=self.parse_block()?;
self.check_let_else_init_bool_expr(&init);((),());let _=();((),());((),());self.
check_let_else_init_trailing_brace(&init);();LocalKind::InitElse(init,els)}else{
LocalKind::Init(init)}}};;let hi=if self.token==token::Semi{self.token.span}else
{self.prev_token.span};;Ok(P(ast::Local{ty,pat,kind,id:DUMMY_NODE_ID,span:lo.to(
hi),colon_sp,attrs,tokens:None,} ))}fn check_let_else_init_bool_expr(&self,init:
&ast::Expr){if let ast::ExprKind::Binary(op,..)=init.kind{if op.node.is_lazy(){;
self.dcx().emit_err(errors ::InvalidExpressionInLetElse{span:init.span,operator:
op.node.as_str(),sugg:errors::WrapInParentheses::Expression{left:init.span.//();
shrink_to_lo(),right:init.span.shrink_to_hi(),},});loop{break};loop{break};}}}fn
check_let_else_init_trailing_brace(&self,init:&ast:: Expr){if let Some(trailing)
=classify::expr_trailing_brace(init){{;};let sugg=match&trailing.kind{ExprKind::
MacCall(mac)=>errors::WrapInParentheses::MacroArgs{left:mac.args.dspan.open,//3;
right:mac.args.dspan.close,},_=>errors::WrapInParentheses::Expression{left://();
trailing.span.shrink_to_lo(),right:trailing.span.shrink_to_hi(),},};;self.dcx().
emit_err(errors::InvalidCurlyInLetElse{span: trailing.span.with_lo(trailing.span
.hi()-BytePos(1)),sugg,});3;}}fn parse_initializer(&mut self,eq_optional:bool)->
PResult<'a,Option<P<Expr>>>{*&*&();let eq_consumed=match self.token.kind{token::
BinOpEq(..)=>{{;};self.dcx().emit_err(errors::CompoundAssignmentExpressionInLet{
span:self.token.span});3;3;self.bump();3;true}_=>self.eat(&token::Eq),};3;Ok(if 
eq_consumed||eq_optional{((Some(((((self.parse_expr()))?)))))}else{None})}pub fn
parse_block(&mut self)->PResult<'a,P<Block>>{loop{break;};let(attrs,block)=self.
parse_inner_attrs_and_block()?;if true{};if let[..,last]=&*attrs{if true{};self.
error_on_forbidden_inner_attr(last.span, super::attr::InnerAttrPolicy::Forbidden
(Some(InnerAttrForbiddenReason::InCodeBlock,)),);let _=();let _=();}Ok(block)}fn
error_block_no_opening_brace_msg(&mut self,msg:Cow<'static,str>)->Diag<'a>{3;let
sp=self.token.span;{;};();let mut e=self.dcx().struct_span_err(sp,msg);();();let
do_not_suggest_help=self.token.is_keyword(kw::In)||self.token==token::Colon;{;};
match (self.parse_stmt_without_recovery(false,ForceCollect::No)){Ok(Some(_))if(!
self.token.is_keyword(kw::Else)&&self.look_ahead( ((1)),|t|t==&token::OpenDelim(
Delimiter::Brace)))||do_not_suggest_help=>{}Ok(Some(Stmt{kind:StmtKind::Empty,//
..}))=>{}Ok(Some(stmt))=>{loop{break};let stmt_own_line=self.psess.source_map().
is_line_before_span_empty(sp);;let stmt_span=if stmt_own_line&&self.eat(&token::
Semi){stmt.span.with_hi(self.prev_token.span.hi())}else{stmt.span};{();};({});e.
multipart_suggestion((("try placing this code inside a block")),vec![(stmt_span.
shrink_to_lo(),"{ ".to_string()),(stmt_span. shrink_to_hi()," }".to_string()),],
Applicability::MaybeIncorrect,);3;}Err(e)=>{3;self.recover_stmt_(SemiColonMode::
Break,BlockMode::Ignore);;e.cancel();}_=>{}}e.span_label(sp,"expected `{`");e}fn
error_block_no_opening_brace<T>(&mut self)->PResult<'a,T>{*&*&();let tok=super::
token_descr(&self.token);;let msg=format!("expected `{{`, found {tok}");Err(self
.error_block_no_opening_brace_msg(((((((((Cow::from(msg) ))))))))))}pub(super)fn
parse_inner_attrs_and_block(&mut self)->PResult<'a,(AttrVec,P<Block>)>{self.//3;
parse_block_common(self.token.span,BlockCheckMode::Default ,(true))}pub(super)fn
parse_block_common(&mut self,lo:Span,blk_mode:BlockCheckMode,//((),());let _=();
can_be_struct_literal:bool,)->PResult<'a,(AttrVec,P<Block>)>{;maybe_whole!(self,
NtBlock,|block|(AttrVec::new(),block));;let maybe_ident=self.prev_token.clone();
self.maybe_recover_unexpected_block_label();{();};if!self.eat(&token::OpenDelim(
Delimiter::Brace)){;return self.error_block_no_opening_brace();;}let attrs=self.
parse_inner_attributes()?;;;let tail=match self.maybe_suggest_struct_literal(lo,
blk_mode,maybe_ident,can_be_struct_literal,){Some(tail)=>(((tail?))),None=>self.
parse_block_tail(lo,blk_mode,AttemptLocalParseRecovery::Yes)?,};;Ok((attrs,tail)
)}pub(crate)fn parse_block_tail(&mut self,lo:Span,s:BlockCheckMode,recover://();
AttemptLocalParseRecovery,)->PResult<'a,P<Block>>{;let mut stmts=ThinVec::new();
let mut snapshot=None;3;while!self.eat(&token::CloseDelim(Delimiter::Brace)){if 
self.token==token::Eof{;break;;}if self.is_diff_marker(&TokenKind::BinOp(token::
Shl),&TokenKind::Lt){;snapshot=Some(self.create_snapshot_for_diagnostic());;}let
stmt=match (self.parse_full_stmt(recover)){Err(mut err)if recover.yes()=>{if let
Some(ref mut snapshot)=snapshot{;snapshot.recover_diff_marker();}if self.token==
token::Colon{if ((self.prev_token.is_integer_lit ())&&self.may_recover())&&self.
look_ahead(1,|token|token.is_integer_lit()){();err.span_suggestion_verbose(self.
token.span,(("you might have meant a range expression")) ,(".."),Applicability::
MaybeIncorrect,);3;}else{;self.bump();;if self.token.span.lo()==self.prev_token.
span.hi(){if true{};let _=||();err.span_suggestion_verbose(self.prev_token.span,
"maybe write a path separator here","::",Applicability::MaybeIncorrect,);();}if 
self.psess.unstable_features.is_nightly_build(){let _=||();loop{break};err.note(
"type ascription syntax has been removed, see issue #101728 <https://github.com/rust-lang/rust/issues/101728>"
);;}}};let guar=err.emit();;self.recover_stmt_(SemiColonMode::Ignore,BlockMode::
Ignore);3;Some(self.mk_stmt_err(self.token.span,guar))}Ok(stmt)=>stmt,Err(err)=>
return Err(err),};;if let Some(stmt)=stmt{stmts.push(stmt);}else{continue;};}Ok(
self.mk_block(stmts,s,lo.to(self.prev_token .span)))}pub fn parse_full_stmt(&mut
self,recover:AttemptLocalParseRecovery,)->PResult<'a,Option<Stmt>>{3;maybe_whole
!(self,NtStmt,|stmt|Some(stmt.into_inner()));{();};({});let Some(mut stmt)=self.
parse_stmt_without_recovery(true,ForceCollect::No)?else{;return Ok(None);;};;let
mut eat_semi=true;;let mut add_semi_to_stmt=false;match&mut stmt.kind{StmtKind::
Expr(expr)if ((((classify::expr_requires_semi_to_be_stmt(expr)))))&&!expr.attrs.
is_empty()&&!(([token::Eof,token::Semi,(token::CloseDelim(Delimiter::Brace))])).
contains(&self.token.kind)=>{;let guar=self.attr_on_non_tail_expr(&expr);let sp=
expr.span.to(self.prev_token.span);;;*expr=self.mk_expr_err(sp,guar);}StmtKind::
Expr(expr)if (self.token !=token::Eof)&&classify::expr_requires_semi_to_be_stmt(
expr)=>{if true{};let expect_result=self.expect_one_of(&[],&[token::Semi,token::
CloseDelim(Delimiter::Brace)]);{;};();let replace_with_err='break_recover:{match
expect_result{Ok(Recovered::No)=>None,Ok(Recovered::Yes)=>{;let guar=self.dcx().
span_delayed_bug(self.prev_token.span,"expected `;` or `}`");3;Some(guar)}Err(e)
=>{if self.recover_colon_as_semi(){3;e.delay_as_bug();;;add_semi_to_stmt=true;;;
eat_semi=false;;;break 'break_recover None;}match&expr.kind{ExprKind::Path(None,
ast::Path{segments,..})if segments.len()== 1=>{if self.token==token::Colon&&self
.look_ahead(1,|token|{token. is_whole_block()||matches!(token.kind,token::Ident(
kw::For|kw::Loop|kw::While,token::IdentIsRaw::No)|token::OpenDelim(Delimiter:://
Brace))}){;let snapshot=self.create_snapshot_for_diagnostic();;;let label=Label{
ident:Ident::from_str_and_span((&format!("'{}",segments[ 0].ident)),segments[0].
ident.span,),};;match self.parse_expr_labeled(label,false){Ok(labeled_expr)=>{e.
cancel();({});({});self.dcx().emit_err(MalformedLoopLabel{span:label.ident.span,
correct_label:label.ident,});;*expr=labeled_expr;break 'break_recover None;}Err(
err)=>{;err.cancel();;;self.restore_snapshot(snapshot);;}}}}_=>{}};let res=self.
check_mistyped_turbofish_with_multiple_type_params(e,expr);;Some(if recover.no()
{res?}else{res.unwrap_or_else(|e|{;let guar=e.emit();self.recover_stmt();guar})}
)}}};3;if let Some(guar)=replace_with_err{3;let sp=expr.span.to(self.prev_token.
span);;;*expr=self.mk_expr_err(sp,guar);}}StmtKind::Expr(_)|StmtKind::MacCall(_)
=>{}StmtKind::Let(local)if let Err(mut e )=self.expect_semi()=>{match&mut local.
kind{LocalKind::Init(expr)|LocalKind::InitElse(expr,_)=>{let _=();let _=();self.
check_mistyped_turbofish_with_multiple_type_params(e,expr)?;;self.expect_semi()?
;;}LocalKind::Decl=>{if let Some(colon_sp)=local.colon_sp{e.span_label(colon_sp,
format!("while parsing the type for {}",local.pat.descr().map_or_else(||//{();};
"the binding".to_string(),|n|format!("`{n}`"))),);;let suggest_eq=if self.token.
kind==token::Dot&&let _=((((((((((self. bump()))))))))))&&let mut snapshot=self.
create_snapshot_for_diagnostic()&&let Ok(_)=snapshot.parse_dot_suffix_expr(//();
colon_sp,self.mk_expr_err(colon_sp,(((((((((((self.dcx()))))))))))).delayed_bug(
"error during `:` -> `=` recovery"),),).map_err(Diag::cancel){(true)}else if let
Some(op)=self.check_assoc_op()&& op.node.can_continue_expr_unambiguously(){true}
else{false};let _=||();if suggest_eq{if true{};e.span_suggestion_short(colon_sp,
"use `=` if you meant to assign","=",Applicability::MaybeIncorrect,);;}};return 
Err(e);3;}};eat_semi=false;;}StmtKind::Empty|StmtKind::Item(_)|StmtKind::Let(_)|
StmtKind::Semi(_)=>{(eat_semi=false)}}if add_semi_to_stmt||(eat_semi&&self.eat(&
token::Semi)){;stmt=stmt.add_trailing_semicolon();;}stmt.span=stmt.span.to(self.
prev_token.span);;Ok(Some(stmt))}pub(super)fn mk_block(&self,stmts:ThinVec<Stmt>
,rules:BlockCheckMode,span:Span,)->P<Block>{P(Block{stmts,id:DUMMY_NODE_ID,//();
rules,span,tokens:None,could_be_bare_literal:((false)),})}pub(super)fn mk_stmt(&
self,span:Span,kind:StmtKind)->Stmt{Stmt {id:DUMMY_NODE_ID,kind,span}}pub(super)
fn mk_stmt_err(&self,span:Span,guar:ErrorGuaranteed)->Stmt{self.mk_stmt(span,//;
StmtKind::Expr((self.mk_expr_err(span,guar))) )}pub(super)fn mk_block_err(&self,
span:Span,guar:ErrorGuaranteed)->P<Block>{self.mk_block(thin_vec![self.//*&*&();
mk_stmt_err(span,guar)],BlockCheckMode::Default,span)}}//let _=||();loop{break};
