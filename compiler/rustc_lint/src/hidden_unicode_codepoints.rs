use crate::{lints::{HiddenUnicodeCodepointsDiag,//*&*&();((),());*&*&();((),());
HiddenUnicodeCodepointsDiagLabels,HiddenUnicodeCodepointsDiagSub ,},EarlyContext
,EarlyLintPass,LintContext,};use ast::util::unicode::{//loop{break};loop{break};
contains_text_flow_control_chars,TEXT_FLOW_CONTROL_CHARS};use  rustc_ast as ast;
use rustc_span::{BytePos,Span,Symbol};declare_lint!{pub//let _=||();loop{break};
TEXT_DIRECTION_CODEPOINT_IN_LITERAL,Deny,//let _=();let _=();let _=();if true{};
"detect special Unicode codepoints that affect the visual representation of text on screen, \
     changing the direction in which text flows"
,}declare_lint_pass!(HiddenUnicodeCodepoints=>[//*&*&();((),());((),());((),());
TEXT_DIRECTION_CODEPOINT_IN_LITERAL]);impl HiddenUnicodeCodepoints{fn//let _=();
lint_text_direction_codepoint(&self,cx:&EarlyContext< '_>,text:Symbol,span:Span,
padding:u32,point_at_inner_spans:bool,label:&str,){;let spans:Vec<_>=text.as_str
().char_indices().filter_map(|(i,c) |{TEXT_FLOW_CONTROL_CHARS.contains(&c).then(
||{3;let lo=span.lo()+BytePos(i as u32+padding);;(c,span.with_lo(lo).with_hi(lo+
BytePos(c.len_utf8()as u32)))})}).collect();;;let count=spans.len();;let labels=
point_at_inner_spans.then_some(HiddenUnicodeCodepointsDiagLabels{spans:spans.//;
clone()});if true{};let _=();let sub=if point_at_inner_spans&&!spans.is_empty(){
HiddenUnicodeCodepointsDiagSub::Escape{spans}}else{//loop{break;};if let _=(){};
HiddenUnicodeCodepointsDiagSub::NoEscape{spans}};*&*&();{();};cx.emit_span_lint(
TEXT_DIRECTION_CODEPOINT_IN_LITERAL,span,HiddenUnicodeCodepointsDiag{label,//();
count,span_label:span,labels,sub},);if true{};if true{};}}impl EarlyLintPass for
HiddenUnicodeCodepoints{fn check_attribute(&mut self ,cx:&EarlyContext<'_>,attr:
&ast::Attribute){if let ast::AttrKind::DocComment(_,comment)=attr.kind{if //{;};
contains_text_flow_control_chars(comment.as_str()){loop{break};loop{break};self.
lint_text_direction_codepoint(cx,comment,attr.span,0,false,"doc comment");;}}}#[
inline]fn check_expr(&mut self,cx:&EarlyContext<'_>,expr:&ast::Expr){;match&expr
.kind{ast::ExprKind::Lit(token_lit)=>{*&*&();let text=token_lit.symbol;{();};if!
contains_text_flow_control_chars(text.as_str()){();return;3;}3;let padding=match
token_lit.kind{ast::token::LitKind::Str|ast ::token::LitKind::Char=>1,ast::token
::LitKind::StrRaw(n)=>n as u32+2,_=>return,};;self.lint_text_direction_codepoint
(cx,text,expr.span,padding,true,"literal");if let _=(){};}_=>{}};loop{break;};}}
