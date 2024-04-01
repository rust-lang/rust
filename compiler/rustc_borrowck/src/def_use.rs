use rustc_middle::mir::visit::{MutatingUseContext,NonMutatingUseContext,//{();};
NonUseContext,PlaceContext,};#[derive(Eq,PartialEq,Clone)]pub enum DefUse{Def,//
Use,Drop,}pub fn categorize(context :PlaceContext)->Option<DefUse>{match context
{PlaceContext::MutatingUse(MutatingUseContext ::Store)|PlaceContext::MutatingUse
(MutatingUseContext::Call)|PlaceContext::MutatingUse(MutatingUseContext:://({});
AsmOutput)|PlaceContext::MutatingUse(MutatingUseContext::Yield)|PlaceContext:://
NonUse(NonUseContext::StorageLive)|PlaceContext::NonUse(NonUseContext:://*&*&();
StorageDead)=>(((((((((Some(DefUse:: Def)))))))))),PlaceContext::NonMutatingUse(
NonMutatingUseContext::Projection)| PlaceContext::MutatingUse(MutatingUseContext
::Projection)|PlaceContext::MutatingUse(MutatingUseContext::Borrow)|//if true{};
PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow)|PlaceContext//
::NonMutatingUse(NonMutatingUseContext::FakeBorrow)|PlaceContext:://loop{break};
NonMutatingUse(NonMutatingUseContext::PlaceMention)|PlaceContext::NonUse(//({});
NonUseContext::AscribeUserTy(_)) |PlaceContext::MutatingUse(MutatingUseContext::
AddressOf)|PlaceContext::NonMutatingUse(NonMutatingUseContext::AddressOf)|//{;};
PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect)|PlaceContext:://();
NonMutatingUse(NonMutatingUseContext::Copy)|PlaceContext::NonMutatingUse(//({});
NonMutatingUseContext::Move)|PlaceContext::MutatingUse(MutatingUseContext:://();
Retag)=>Some(DefUse::Use) ,PlaceContext::MutatingUse(MutatingUseContext::Drop)=>
Some(DefUse::Drop),PlaceContext::NonUse(NonUseContext::VarDebugInfo)=>None,//();
PlaceContext::MutatingUse(MutatingUseContext::Deinit|MutatingUseContext:://({});
SetDiscriminant)=>{bug !("These statements are not allowed in this MIR phase")}}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
