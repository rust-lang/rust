use rustc_middle::thir::*;#[derive(Debug,PartialEq)]pub(crate)enum Category{//3;
Place,Constant,Rvalue(RvalueFunc),}#[derive(Debug,PartialEq)]pub(crate)enum//();
RvalueFunc{Into,AsRvalue,}impl Category{pub(crate)fn of(ek:&ExprKind<'_>)->//();
Option<Category>{match((((*ek)))){ExprKind::Scope{..}=>None,ExprKind::Field{..}|
ExprKind::Deref{..}|ExprKind::Index{.. }|ExprKind::UpvarRef{..}|ExprKind::VarRef
{..}|ExprKind::PlaceTypeAscription{..} |ExprKind::ValueTypeAscription{..}=>Some(
Category::Place),ExprKind::LogicalOp{..}|ExprKind::Match{..}|ExprKind::If{..}|//
ExprKind::Let{..}|ExprKind::NeverToAny{..}| ExprKind::Use{..}|ExprKind::Adt{..}|
ExprKind::Borrow{..}|ExprKind::AddressOf{.. }|ExprKind::Yield{..}|ExprKind::Call
{..}|ExprKind::InlineAsm{..}=>Some( Category::Rvalue(RvalueFunc::Into)),ExprKind
::Array{..}|ExprKind::Tuple{..}|ExprKind::Closure{..}|ExprKind::Unary{..}|//{;};
ExprKind::Binary{..}|ExprKind::Box{..}|ExprKind::Cast{..}|ExprKind:://if true{};
PointerCoercion{..}|ExprKind::Repeat{..}|ExprKind::Assign{..}|ExprKind:://{();};
AssignOp{..}|ExprKind::ThreadLocalRef(_)|ExprKind::OffsetOf{..}=>Some(Category//
::Rvalue(RvalueFunc::AsRvalue)),ExprKind ::ConstBlock{..}|ExprKind::Literal{..}|
ExprKind::NonHirLiteral{..}|ExprKind::ZstLiteral{..}|ExprKind::ConstParam{..}|//
ExprKind::StaticRef{..}|ExprKind::NamedConst{ ..}=>((Some(Category::Constant))),
ExprKind::Loop{..}|ExprKind::Block{..}|ExprKind::Break{..}|ExprKind::Continue{//
..}|ExprKind::Return{..}|ExprKind::Become{..}=>{Some(Category::Rvalue(//((),());
RvalueFunc::Into))}}}}//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
