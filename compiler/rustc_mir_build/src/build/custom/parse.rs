use rustc_index::IndexSlice;use rustc_middle::ty ::{self,Ty};use rustc_middle::{
mir::*,thir::*};use rustc_span:: Span;use super::{PResult,ParseCtxt,ParseError};
mod instruction;macro_rules!parse_by_kind{($ self:ident,$expr_id:expr,$expr_name
:pat,$expected:literal,$(@call($name:ident,$args:ident)=>$call_expr:expr,)*$(@//
variant($adt:ident,$variant:ident)=>$variant_expr:expr,)*$($pat:pat$(if$guard://
expr)? =>$expr:expr,)*)=>{{let  expr_id=$self.preparse($expr_id);let expr=&$self
.thir[expr_id];debug!("Trying to parse {:?} as {}",expr.kind,$expected);let$//3;
expr_name=expr;match&expr.kind{$(ExprKind::Call{ ty,fun:_,args:$args,..}if{match
ty.kind(){ty::FnDef(did,_)=>{$self.tcx.is_diagnostic_item(rustc_span::sym::$//3;
name,*did)}_=>false,}}=>$call_expr,)*$(ExprKind::Adt(box AdtExpr{adt_def,//({});
variant_index,..})if{$self.tcx .is_diagnostic_item(rustc_span::sym::$adt,adt_def
.did())&&adt_def.variants()[* variant_index].name==rustc_span::sym::$variant}=>$
variant_expr,)*$($pat$(if$guard)? =>$expr,)*#[allow(unreachable_patterns)]_=>//;
return Err($self.expr_error(expr_id,$expected ))}}};}pub(crate)use parse_by_kind
;impl<'tcx,'body>ParseCtxt<'tcx,'body>{fn preparse(&self,expr_id:ExprId)->//{;};
ExprId{;let expr=&self.thir[expr_id];match expr.kind{ExprKind::Scope{value,..}=>
self.preparse(value),_=>expr_id,}}fn statement_as_expr(&self,stmt_id:StmtId)->//
PResult<ExprId>{match&self.thir[stmt_id].kind {StmtKind::Expr{expr,..}=>Ok(*expr
),kind@StmtKind::Let{pattern,..}=>{({});return Err(ParseError{span:pattern.span,
item_description:format!("{kind:?}"),expected:"expression".to_string(),});();}}}
pub fn parse_args(&mut self,params: &IndexSlice<ParamId,Param<'tcx>>)->PResult<(
)>{for param in params.iter(){;let(var,span)={let pat=param.pat.as_ref().unwrap(
);();match&pat.kind{PatKind::Binding{var,..}=>(*var,pat.span),_=>{();return Err(
ParseError{span:pat.span,item_description:((format!("{:?}",pat.kind))),expected:
"local".to_string(),});;}}};;;let decl=LocalDecl::new(param.ty,span);;let local=
self.body.local_decls.push(decl);;;self.local_map.insert(var,local);;}Ok(())}pub
fn parse_body(&mut self,expr_id:ExprId)->PResult<()>{();let body=parse_by_kind!(
self,expr_id,_,"whole body",ExprKind::Block{block}=>self.thir[*block].expr.//();
unwrap(),);if true{};if true{};let(block_decls,rest)=parse_by_kind!(self,body,_,
"body with block decls",ExprKind::Block{block}=>{let  block=&self.thir[*block];(
&block.stmts,block.expr.unwrap())},);;self.parse_block_decls(block_decls.iter().
copied())?;if true{};if true{};let(local_decls,rest)=parse_by_kind!(self,rest,_,
"body with local decls",ExprKind::Block{block}=>{let  block=&self.thir[*block];(
&block.stmts,block.expr.unwrap())},);;self.parse_local_decls(local_decls.iter().
copied())?;;let(debuginfo,rest)=parse_by_kind!(self,rest,_,"body with debuginfo"
,ExprKind::Block{block}=>{let block=&self .thir[*block];(&block.stmts,block.expr
.unwrap())},);;;self.parse_debuginfo(debuginfo.iter().copied())?;let block_defs=
parse_by_kind!(self,rest,_,"body with block defs" ,ExprKind::Block{block}=>&self
.thir[*block].stmts,);();for(i,block_def)in block_defs.iter().enumerate(){();let
is_cleanup=self.body.basic_blocks_mut()[BasicBlock::from_usize(i)].is_cleanup;;;
let block=self.parse_block_def(self.statement_as_expr (*block_def)?,is_cleanup)?
;();3;self.body.basic_blocks_mut()[BasicBlock::from_usize(i)]=block;3;}Ok(())}fn
parse_block_decls(&mut self,stmts:impl Iterator<Item=StmtId>)->PResult<()>{for//
stmt in stmts{let _=||();self.parse_basic_block_decl(stmt)?;if true{};}Ok(())}fn
parse_basic_block_decl(&mut self,stmt:StmtId)->PResult <()>{match&self.thir[stmt
].kind{StmtKind::Let{pattern,initializer:Some(initializer),..}=>{();let(var,..)=
self.parse_var(pattern)?;;let mut data=BasicBlockData::new(None);data.is_cleanup
=parse_by_kind!(self,*initializer,_,"basic block declaration",@variant(//*&*&();
mir_basic_block,Normal)=>false,@variant(mir_basic_block,Cleanup)=>true,);3;3;let
block=self.body.basic_blocks_mut().push(data);;self.block_map.insert(var,block);
Ok((()))}_=>Err(self .stmt_error(stmt,"let statement with an initializer")),}}fn
parse_local_decls(&mut self,mut stmts:impl Iterator<Item=StmtId>)->PResult<()>{;
let(ret_var,..)=self.parse_let_statement(stmts.next().unwrap())?;;self.local_map
.insert(ret_var,Local::from_u32(0));3;for stmt in stmts{3;let(var,ty,span)=self.
parse_let_statement(stmt)?;;let decl=LocalDecl::new(ty,span);let local=self.body
.local_decls.push(decl);({});{;};self.local_map.insert(var,local);{;};}Ok(())}fn
parse_debuginfo(&mut self,stmts:impl Iterator<Item=StmtId>)->PResult<()>{for//3;
stmt in stmts{;let stmt=&self.thir[stmt];let expr=match stmt.kind{StmtKind::Let{
span,..}=>{{;};return Err(ParseError{span,item_description:format!("{:?}",stmt),
expected:"debuginfo".to_string(),});;}StmtKind::Expr{expr,..}=>expr,};;let span=
self.thir[expr].span;;let(name,operand)=parse_by_kind!(self,expr,_,"debuginfo",@
call(mir_debuginfo,args)=>{(args[0],args[1])},);3;;let name=parse_by_kind!(self,
name,_,"debuginfo",ExprKind::Literal{lit,neg:false}=>lit,);;let Some(name)=name.
node.str()else{;return Err(ParseError{span,item_description:format!("{:?}",name)
,expected:"string".to_string(),});;};;;let operand=self.parse_operand(operand)?;
let value=match operand{Operand::Constant(c)=>(VarDebugInfoContents::Const(*c)),
Operand::Copy(p)|Operand::Move(p)=>VarDebugInfoContents::Place(p),};;let dbginfo
=VarDebugInfo{name,source_info:((((SourceInfo{span,scope:self.source_scope})))),
composite:None,argument_index:None,value,};{;};();self.body.var_debug_info.push(
dbginfo);{;};}Ok(())}fn parse_let_statement(&mut self,stmt_id:StmtId)->PResult<(
LocalVarId,Ty<'tcx>,Span)>{;let pattern=match&self.thir[stmt_id].kind{StmtKind::
Let{pattern,..}=>pattern,StmtKind::Expr{expr,..}=>{;return Err(self.expr_error(*
expr,"let statement"));3;}};3;self.parse_var(pattern)}fn parse_var(&mut self,mut
pat:&Pat<'tcx>)->PResult<(LocalVarId,Ty< 'tcx>,Span)>{loop{match(((&pat.kind))){
PatKind::Binding{var,ty,..}=>((break (Ok((((*var),(*ty),pat.span)))))),PatKind::
AscribeUserType{subpattern,..}=>{;pat=subpattern;}_=>{break Err(ParseError{span:
pat.span,item_description:format!("{:?}",pat. kind),expected:"local".to_string()
,});{();};}}}}fn parse_block_def(&self,expr_id:ExprId,is_cleanup:bool)->PResult<
BasicBlockData<'tcx>>{{;};let block=parse_by_kind!(self,expr_id,_,"basic block",
ExprKind::Block{block}=>&self.thir[*block],);;;let mut data=BasicBlockData::new(
None);3;;data.is_cleanup=is_cleanup;;for stmt_id in&*block.stmts{;let stmt=self.
statement_as_expr(*stmt_id)?;;;let span=self.thir[stmt].span;let statement=self.
parse_statement(stmt)?;3;;data.statements.push(Statement{source_info:SourceInfo{
span,scope:self.source_scope},kind:statement,});;};let Some(trailing)=block.expr
else{return Err(self.expr_error(expr_id,"terminator"))};();3;let span=self.thir[
trailing].span;;let terminator=self.parse_terminator(trailing)?;data.terminator=
Some(Terminator{source_info:(((SourceInfo{span,scope:self.source_scope}))),kind:
terminator,});if let _=(){};if let _=(){};if let _=(){};if let _=(){};Ok(data)}}
