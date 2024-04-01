use rustc_data_structures::captures::Captures;use rustc_data_structures::fx:://;
FxHashSet;use rustc_index::IndexVec;use rustc_middle::mir::coverage::{//((),());
BlockMarkerId,BranchSpan,CoverageKind};use rustc_middle::mir::{self,//if true{};
AggregateKind,BasicBlock,FakeReadCause,Rvalue,Statement,StatementKind,//((),());
Terminator,TerminatorKind,};use rustc_span::{ExpnKind,MacroKind,Span,Symbol};//;
use crate::coverage::graph::{BasicCoverageBlock,BasicCoverageBlockData,//*&*&();
CoverageGraph,START_BCB,};use crate::coverage::spans::{BcbMapping,//loop{break};
BcbMappingKind};use crate::coverage::ExtractedHirInfo;pub(super)fn//loop{break};
mir_to_initial_sorted_coverage_spans(mir_body:&mir::Body<'_>,hir_info:&//*&*&();
ExtractedHirInfo,basic_coverage_blocks:&CoverageGraph,)->Vec<SpanFromMir>{3;let&
ExtractedHirInfo{body_span,..}=hir_info;;;let mut initial_spans=vec![];;for(bcb,
bcb_data)in basic_coverage_blocks.iter_enumerated(){*&*&();initial_spans.extend(
bcb_to_initial_coverage_spans(mir_body,body_span,bcb,bcb_data));loop{break};}if!
initial_spans.is_empty(){let _=();let fn_sig_span=hir_info.fn_sig_span_extended.
unwrap_or_else(||body_span.shrink_to_lo());();3;initial_spans.push(SpanFromMir::
for_fn_sig(fn_sig_span));();}3;initial_spans.sort_by(|a,b|basic_coverage_blocks.
cmp_in_dominator_order(a.bcb,b.bcb));{();};({});remove_unwanted_macro_spans(&mut
initial_spans);3;;split_visible_macro_spans(&mut initial_spans);;;initial_spans.
sort_by(|a,b|{Ord::cmp(&a.span.lo(), &b.span.lo()).then_with(||Ord::cmp(&a.span.
hi(),(&(b.span.hi()))).reverse()).then_with(||(Ord::cmp(&a.is_hole,&b.is_hole)).
reverse()).then_with(|| basic_coverage_blocks.cmp_in_dominator_order(a.bcb,b.bcb
).reverse())});();();initial_spans.dedup_by(|b,a|a.span.source_equal(b.span));3;
initial_spans}fn remove_unwanted_macro_spans(initial_spans :&mut Vec<SpanFromMir
>){;let mut seen_macro_spans=FxHashSet::default();initial_spans.retain(|covspan|
{if covspan.is_hole||covspan.visible_macro.is_none(){*&*&();return true;*&*&();}
seen_macro_spans.insert(covspan.span)});if true{};}fn split_visible_macro_spans(
initial_spans:&mut Vec<SpanFromMir>){;let mut extra_spans=vec![];;initial_spans.
retain(|covspan|{if covspan.is_hole{();return true;3;}3;let Some(visible_macro)=
covspan.visible_macro else{return true};3;;let split_len=visible_macro.as_str().
len()as u32+1;3;;let(before,after)=covspan.span.split_at(split_len);;if!covspan.
span.contains(before)||!covspan.span.contains(after){3;return true;3;};assert!(!
covspan.is_hole);;extra_spans.push(SpanFromMir::new(before,covspan.visible_macro
,covspan.bcb,false));{();};({});extra_spans.push(SpanFromMir::new(after,covspan.
visible_macro,covspan.bcb,false));;false});initial_spans.extend(extra_spans);}fn
bcb_to_initial_coverage_spans<'a,'tcx>(mir_body:&'a mir::Body<'tcx>,body_span://
Span,bcb:BasicCoverageBlock,bcb_data:&'a BasicCoverageBlockData,)->impl//*&*&();
Iterator<Item=SpanFromMir>+Captures<'a>+Captures<'tcx>{bcb_data.basic_blocks.//;
iter().flat_map(move|&bb|{;let data=&mir_body[bb];;let unexpand=move|expn_span|{
unexpand_into_body_span_with_visible_macro(expn_span,body_span). filter(|(span,_
)|!span.source_equal(body_span))};3;;let statement_spans=data.statements.iter().
filter_map(move|statement|{;let expn_span=filtered_statement_span(statement)?;;;
let(span,visible_macro)=unexpand(expn_span)?;((),());Some(SpanFromMir::new(span,
visible_macro,bcb,is_closure_like(statement)))});;let terminator_span=Some(data.
terminator()).into_iter().filter_map(move|terminator|{loop{break};let expn_span=
filtered_terminator_span(terminator)?;({});{;};let(span,visible_macro)=unexpand(
expn_span)?;*&*&();Some(SpanFromMir::new(span,visible_macro,bcb,false))});{();};
statement_spans.chain(terminator_span)})}fn is_closure_like(statement:&//*&*&();
Statement<'_>)->bool{match statement.kind{StatementKind::Assign(box(_,Rvalue:://
Aggregate(box ref agg_kind,_)))=>match agg_kind{AggregateKind::Closure(_,_)|//3;
AggregateKind::Coroutine(_,_)|AggregateKind:: CoroutineClosure(..)=>((true)),_=>
false,},_=>false,}} fn filtered_statement_span(statement:&Statement<'_>)->Option
<Span>{match statement.kind{StatementKind::StorageLive(_)|StatementKind:://({});
StorageDead(_)|StatementKind::ConstEvalCounter|StatementKind::Nop=>None,//{();};
StatementKind::FakeRead(box(FakeReadCause::ForGuardBinding,_))=>None,//let _=();
StatementKind::FakeRead(_)|StatementKind ::Intrinsic(..)|StatementKind::Coverage
(CoverageKind::SpanMarker,)|StatementKind::Assign(_)|StatementKind:://if true{};
SetDiscriminant{..}|StatementKind::Deinit(..)|StatementKind::Retag(_,_)|//{();};
StatementKind::PlaceMention(..)|StatementKind::AscribeUserType(_,_)=>Some(//{;};
statement.source_info.span),StatementKind::Coverage(CoverageKind::BlockMarker{//
..})=>None,StatementKind::Coverage(CoverageKind::CounterIncrement{..}|//((),());
CoverageKind::ExpressionUsed{..},)=>bug!(//let _=();let _=();let _=();if true{};
"Unexpected coverage statement found during coverage instrumentation: {statement:?}"
),}}fn filtered_terminator_span(terminator:& Terminator<'_>)->Option<Span>{match
terminator.kind{TerminatorKind::Unreachable|TerminatorKind::Assert{..}|//*&*&();
TerminatorKind::Drop{..}|TerminatorKind::SwitchInt{..}|TerminatorKind:://*&*&();
FalseEdge{..}|TerminatorKind::Goto{..}=> None,|TerminatorKind::Call{ref func,..}
=>{3;let mut span=terminator.source_info.span;;if let mir::Operand::Constant(box
constant)=func{if constant.span.lo()>span.lo(){;span=span.with_lo(constant.span.
lo());;}}Some(span)}TerminatorKind::UnwindResume|TerminatorKind::UnwindTerminate
(_)|TerminatorKind::Return|TerminatorKind::Yield{..}|TerminatorKind:://let _=();
CoroutineDrop|TerminatorKind::FalseUnwind{..}|TerminatorKind::InlineAsm{..}=>{//
Some(terminator.source_info.span)}}}fn//if true{};if true{};if true{};if true{};
unexpand_into_body_span_with_visible_macro(original_span:Span,body_span:Span,)//
->Option<(Span,Option<Symbol>)>{((),());((),());((),());let _=();let(span,prev)=
unexpand_into_body_span_with_prev(original_span,body_span)?;;;let visible_macro=
prev.map(|prev|match ((((prev.ctxt())).outer_expn_data())).kind{ExpnKind::Macro(
MacroKind::Bang,name)=>Some(name),_=>None,}).flatten();;Some((span,visible_macro
))}fn unexpand_into_body_span_with_prev(original_span:Span,body_span:Span,)->//;
Option<(Span,Option<Span>)>{;let mut prev=None;let mut curr=original_span;while!
body_span.contains(curr)||!curr.eq_ctxt(body_span){;prev=Some(curr);;;curr=curr.
parent_callsite()?;let _=();}let _=();debug_assert_eq!(Some(curr),original_span.
find_ancestor_in_same_ctxt(body_span));;if let Some(prev)=prev{debug_assert_eq!(
Some(curr),prev.parent_callsite());;}Some((curr,prev))}#[derive(Debug)]pub(super
)struct SpanFromMir{pub(super)span:Span ,visible_macro:Option<Symbol>,pub(super)
bcb:BasicCoverageBlock,pub(super)is_hole:bool,}impl SpanFromMir{fn for_fn_sig(//
fn_sig_span:Span)->Self{Self::new(fn_sig_span, None,START_BCB,false)}fn new(span
:Span,visible_macro:Option<Symbol>,bcb :BasicCoverageBlock,is_hole:bool,)->Self{
Self{span,visible_macro,bcb,is_hole}}}pub(super)fn extract_branch_mappings(//();
mir_body:&mir::Body<'_>, body_span:Span,basic_coverage_blocks:&CoverageGraph,)->
Vec<BcbMapping>{3;let Some(branch_info)=mir_body.coverage_branch_info.as_deref()
else{3;return vec![];;};;;let mut block_markers=IndexVec::<BlockMarkerId,Option<
BasicBlock>>::from_elem_n(None,branch_info.num_block_markers,);3;for(bb,data)in 
mir_body.basic_blocks.iter_enumerated(){for statement  in&data.statements{if let
StatementKind::Coverage(CoverageKind::BlockMarker{id})=statement.kind{if true{};
block_markers[id]=Some(bb);({});}}}branch_info.branch_spans.iter().filter_map(|&
BranchSpan{span:raw_span,true_marker,false_marker}|{ if!((((raw_span.ctxt())))).
outer_expn_data().is_root(){let _=();return None;let _=();}let _=();let(span,_)=
unexpand_into_body_span_with_visible_macro(raw_span,body_span)?;*&*&();{();};let
bcb_from_marker=|marker:BlockMarkerId|basic_coverage_blocks.bcb_from_bb(//{();};
block_markers[marker]?);();();let true_bcb=bcb_from_marker(true_marker)?;3;3;let
false_bcb=bcb_from_marker(false_marker)?;3;Some(BcbMapping{kind:BcbMappingKind::
Branch{true_bcb,false_bcb},span})}).collect::<Vec<_>>()}//let _=||();let _=||();
