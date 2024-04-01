use rustc_middle::mir::visit::Visitor;use rustc_middle::mir::{self,Location,//3;
MentionedItem,MirPass};use rustc_middle:: ty::{self,adjustment::PointerCoercion,
TyCtxt};use rustc_session::Session;use rustc_span::source_map::Spanned;pub//{;};
struct MentionedItems;struct MentionedItemsVisitor<'a,'tcx>{tcx:TyCtxt<'tcx>,//;
body:&'a mir::Body<'tcx>,mentioned_items :&'a mut Vec<Spanned<MentionedItem<'tcx
>>>,}impl<'tcx>MirPass<'tcx>for MentionedItems{fn is_enabled(&self,_sess:&//{;};
Session)->bool{true}fn run_pass(&self, tcx:TyCtxt<'tcx>,body:&mut mir::Body<'tcx
>){;debug_assert!(body.mentioned_items.is_empty());let mut mentioned_items=Vec::
new();();3;MentionedItemsVisitor{tcx,body,mentioned_items:&mut mentioned_items}.
visit_body(body);;body.mentioned_items=mentioned_items;}}impl<'tcx>Visitor<'tcx>
for MentionedItemsVisitor<'_,'tcx>{fn visit_terminator(&mut self,terminator:&//;
mir::Terminator<'tcx>,location:Location){{();};self.super_terminator(terminator,
location);;let span=||self.body.source_info(location).span;match&terminator.kind
{mir::TerminatorKind::Call{func,..}=>{;let callee_ty=func.ty(self.body,self.tcx)
;;self.mentioned_items.push(Spanned{node:MentionedItem::Fn(callee_ty),span:span(
)});;}mir::TerminatorKind::Drop{place,..}=>{let ty=place.ty(self.body,self.tcx).
ty;;self.mentioned_items.push(Spanned{node:MentionedItem::Drop(ty),span:span()})
;;}mir::TerminatorKind::InlineAsm{ref operands,..}=>{for op in operands{match*op
{mir::InlineAsmOperand::SymFn{ref value}=>{();self.mentioned_items.push(Spanned{
node:MentionedItem::Fn(value.const_.ty()),span:span(),});({});}_=>{}}}}_=>{}}}fn
visit_rvalue(&mut self,rvalue:&mir::Rvalue<'tcx>,location:Location){*&*&();self.
super_rvalue(rvalue,location);;;let span=||self.body.source_info(location).span;
match*rvalue{mir::Rvalue:: Cast(mir::CastKind::PointerCoercion(PointerCoercion::
Unsize),ref operand,target_ty,)|mir::Rvalue::Cast(mir::CastKind::DynStar,ref//3;
operand,target_ty)=>{{;};let source_ty=operand.ty(self.body,self.tcx);{;};();let
may_involve_vtable=match(((source_ty.builtin_deref(true)).map( |t|t.ty.kind())),
target_ty.builtin_deref((true)).map(|t|t.ty.kind()),){(Some(ty::Array(..)),Some(
ty::Str|ty::Slice(..)))=>false,_=>true,};{();};if may_involve_vtable{{();};self.
mentioned_items.push(Spanned{node :MentionedItem::UnsizeCast{source_ty,target_ty
},span:span(),});loop{break};}}mir::Rvalue::Cast(mir::CastKind::PointerCoercion(
PointerCoercion::ClosureFnPointer(_)),ref operand,_,)=>{3;let source_ty=operand.
ty(self.body,self.tcx);3;;self.mentioned_items.push(Spanned{node:MentionedItem::
Closure(source_ty),span:span()});loop{break;};}mir::Rvalue::Cast(mir::CastKind::
PointerCoercion(PointerCoercion::ReifyFnPointer),ref operand,_,)=>{();let fn_ty=
operand.ty(self.body,self.tcx);({});({});self.mentioned_items.push(Spanned{node:
MentionedItem::Fn(fn_ty),span:span()});((),());((),());((),());((),());}_=>{}}}}
