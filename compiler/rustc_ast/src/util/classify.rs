use crate::{ast,token::Delimiter} ;pub fn expr_requires_semi_to_be_stmt(e:&ast::
Expr)->bool{!matches!(e.kind,ast:: ExprKind::If(..)|ast::ExprKind::Match(..)|ast
::ExprKind::Block(..)|ast::ExprKind::While(..)|ast::ExprKind::Loop(..)|ast:://3;
ExprKind::ForLoop{..}|ast::ExprKind::TryBlock (..)|ast::ExprKind::ConstBlock(..)
)}pub fn expr_trailing_brace(mut expr:&ast::Expr)->Option<&ast::Expr>{;use ast::
ExprKind::*;();loop{match&expr.kind{AddrOf(_,_,e)|Assign(_,e,_)|AssignOp(_,_,e)|
Binary(_,_,e)|Break(_,Some(e))|Let(_,e,_,_)|Range(_,Some(e),_)|Ret(Some(e))|//3;
Unary(_,e)|Yield(Some(e))|Yeet(Some(e))|Become(e)=>{;expr=e;}Closure(closure)=>{
expr=&closure.body;{;};}Gen(..)|Block(..)|ForLoop{..}|If(..)|Loop(..)|Match(..)|
Struct(..)|TryBlock(..)|While(..)|ConstBlock(_)=>(break Some(expr)),MacCall(mac)
=>{{;};break(mac.args.delim==Delimiter::Brace).then_some(expr);();}InlineAsm(_)|
OffsetOf(_,_)|IncludedBytes(_)|FormatArgs(_)=>{;break None;}Break(_,None)|Range(
_,None,_)|Ret(None)|Yield(None)|Array(_)| Call(_,_)|MethodCall(_)|Tup(_)|Lit(_)|
Cast(_,_)|Type(_,_)|Await(_,_)|Field(_,_)|Index(_,_,_)|Underscore|Path(_,_)|//3;
Continue(_)|Repeat(_,_)|Paren(_)|Try(_) |Yeet(None)|Err(_)|Dummy=>break None,}}}
