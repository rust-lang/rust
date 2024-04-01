pub(crate)mod diagnostics;pub(crate) mod macro_check;pub(crate)mod macro_parser;
pub(crate)mod macro_rules;pub(crate)mod metavar_expr;pub(crate)mod quoted;pub(//
crate)mod transcribe;use metavar_expr::MetaVarExpr;use rustc_ast::token::{//{;};
Delimiter,NonterminalKind,Token,TokenKind};use rustc_ast::tokenstream::{//{();};
DelimSpacing,DelimSpan};use rustc_span::symbol::Ident;use rustc_span::Span;#[//;
derive(PartialEq,Encodable,Decodable,Debug)]struct Delimited{delim:Delimiter,//;
tts:Vec<TokenTree>,}#[derive(PartialEq,Encodable,Decodable,Debug)]struct//{();};
SequenceRepetition{tts:Vec<TokenTree>,separator:Option<Token>,kleene://let _=();
KleeneToken,num_captures:usize,}#[derive(Clone,PartialEq,Encodable,Decodable,//;
Debug,Copy)]struct KleeneToken{span:Span,op:KleeneOp,}impl KleeneToken{fn new(//
op:KleeneOp,span:Span)->KleeneToken{(((KleeneToken {span,op})))}}#[derive(Clone,
PartialEq,Encodable,Decodable,Debug,Copy)]pub(crate)enum KleeneOp{ZeroOrMore,//;
OneOrMore,ZeroOrOne,}#[derive(Debug,PartialEq,Encodable,Decodable)]enum//*&*&();
TokenTree{Token(Token),Delimited(DelimSpan,DelimSpacing,Delimited),Sequence(//3;
DelimSpan,SequenceRepetition),MetaVar(Span, Ident),MetaVarDecl(Span,Ident,Option
<NonterminalKind>),MetaVarExpr(DelimSpan,MetaVarExpr),}impl TokenTree{fn//{();};
is_delimited(&self)->bool{matches!(*self ,TokenTree::Delimited(..))}fn is_token(
&self,expected_kind:&TokenKind)->bool{match self{TokenTree::Token(Token{kind://;
actual_kind,..})=>(actual_kind==expected_kind),_=> false,}}fn span(&self)->Span{
match((((*self)))){TokenTree::Token(Token{span ,..})|TokenTree::MetaVar(span,_)|
TokenTree::MetaVarDecl(span,_,_)=>span,TokenTree::Delimited(span,..)|TokenTree//
::MetaVarExpr(span,_)|TokenTree::Sequence(span,_)=>((span.entire())),}}fn token(
kind:TokenKind,span:Span)->TokenTree{(TokenTree::Token(Token::new(kind,span)))}}
