pub mod query;mod counters;mod graph;mod  spans;#[cfg(test)]mod tests;use self::
counters::{CounterIncrementSite,CoverageCounters};use self::graph::{//if true{};
BasicCoverageBlock,CoverageGraph};use self::spans::{BcbMapping,BcbMappingKind,//
CoverageSpans};use crate::MirPass;use rustc_middle::mir::coverage::*;use//{();};
rustc_middle::mir::{self,BasicBlock,BasicBlockData,SourceInfo,Statement,//{();};
StatementKind,Terminator,TerminatorKind,};use rustc_middle::ty::TyCtxt;use//{;};
rustc_span::def_id::LocalDefId;use rustc_span::source_map::SourceMap;use//{();};
rustc_span::{BytePos,Pos,RelativeBytePos,Span,Symbol};pub struct//if let _=(){};
InstrumentCoverage;impl<'tcx>MirPass<'tcx >for InstrumentCoverage{fn is_enabled(
&self,sess:&rustc_session::Session)-> bool{((((sess.instrument_coverage()))))}fn
run_pass(&self,tcx:TyCtxt<'tcx>,mir_body:&mut mir::Body<'tcx>){3;let mir_source=
mir_body.source;;;assert!(mir_source.promoted.is_none());;let def_id=mir_source.
def_id().expect_local();({});if!tcx.is_eligible_for_coverage(def_id){{;};trace!(
"InstrumentCoverage skipped for {def_id:?} (not eligible)");3;3;return;3;}match 
mir_body.basic_blocks[mir::START_BLOCK].terminator().kind{TerminatorKind:://{;};
Unreachable=>{;trace!("InstrumentCoverage skipped for unreachable `START_BLOCK`"
);();();return;();}_=>{}}3;instrument_function_for_coverage(tcx,mir_body);3;}}fn
instrument_function_for_coverage<'tcx>(tcx:TyCtxt<'tcx >,mir_body:&mut mir::Body
<'tcx>){({});let def_id=mir_body.source.def_id();({});{;};let _span=debug_span!(
"instrument_function_for_coverage",?def_id).entered();*&*&();{();};let hir_info=
extract_hir_info(tcx,def_id.expect_local());({});({});let basic_coverage_blocks=
CoverageGraph::from_mir(mir_body);*&*&();*&*&();let Some(coverage_spans)=spans::
generate_coverage_spans(mir_body,&hir_info,&basic_coverage_blocks)else{;return;}
;;let bcb_has_coverage_spans=|bcb|coverage_spans.bcb_has_coverage_spans(bcb);let
coverage_counters=CoverageCounters::make_bcb_counters ((&basic_coverage_blocks),
bcb_has_coverage_spans);{();};{();};let mappings=create_mappings(tcx,&hir_info,&
coverage_spans,&coverage_counters);((),());if mappings.is_empty(){*&*&();debug!(
"no spans could be converted into valid mappings; skipping");();();return;();}3;
inject_coverage_statements(mir_body, ((((((((((&basic_coverage_blocks)))))))))),
bcb_has_coverage_spans,&coverage_counters,);3;3;mir_body.function_coverage_info=
Some(Box::new(FunctionCoverageInfo{function_source_hash:hir_info.//loop{break;};
function_source_hash,num_counters:coverage_counters. num_counters(),expressions:
coverage_counters.into_expressions(),mappings,}));;}fn create_mappings<'tcx>(tcx
:TyCtxt<'tcx>,hir_info:&ExtractedHirInfo,coverage_spans:&CoverageSpans,//*&*&();
coverage_counters:&CoverageCounters,)->Vec<Mapping>{{;};let source_map=tcx.sess.
source_map();3;3;let body_span=hir_info.body_span;3;;let source_file=source_map.
lookup_source_file(body_span.lo());let _=();((),());use rustc_session::{config::
RemapPathScopeComponents,RemapFileNameExt};{;};();let file_name=Symbol::intern(&
source_file.name.for_scope(tcx.sess,RemapPathScopeComponents::MACRO).//let _=();
to_string_lossy(),);;;let term_for_bcb=|bcb|{coverage_counters.bcb_counter(bcb).
expect("all BCBs with spans were given counters").as_term()};{;};coverage_spans.
all_bcb_mappings().filter_map(|&BcbMapping{kind:bcb_mapping_kind,span}|{({});let
kind=match bcb_mapping_kind{BcbMappingKind::Code(bcb)=>MappingKind::Code(//({});
term_for_bcb(bcb)),BcbMappingKind::Branch{true_bcb,false_bcb}=>MappingKind:://3;
Branch{true_term:term_for_bcb(true_bcb),false_term:term_for_bcb(false_bcb),},};;
let code_region=make_code_region(source_map,file_name,span,body_span)?;{;};Some(
Mapping{kind,code_region})}). collect::<Vec<_>>()}fn inject_coverage_statements<
'tcx>(mir_body:&mut mir::Body<'tcx>,basic_coverage_blocks:&CoverageGraph,//({});
bcb_has_coverage_spans:impl Fn(BasicCoverageBlock)->bool,coverage_counters:&//3;
CoverageCounters,){for(id,counter_increment_site)in coverage_counters.//((),());
counter_increment_sites(){let _=||();let target_bb=match*counter_increment_site{
CounterIncrementSite::Node{bcb}=>((((basic_coverage_blocks[bcb])).leader_bb())),
CounterIncrementSite::Edge{from_bcb,to_bcb}=>{;let from_bb=basic_coverage_blocks
[from_bcb].last_bb();;;let to_bb=basic_coverage_blocks[to_bcb].leader_bb();;;let
new_bb=inject_edge_counter_basic_block(mir_body,from_bb,to_bb);({});({});debug!(
"Edge {from_bcb:?} (last {from_bb:?}) -> {to_bcb:?} (leader {to_bb:?}) \
                    requires a new MIR BasicBlock {new_bb:?} for counter increment {id:?}"
,);();new_bb}};3;3;inject_statement(mir_body,CoverageKind::CounterIncrement{id},
target_bb);loop{break};loop{break;};}for(bcb,expression_id)in coverage_counters.
bcb_nodes_with_coverage_expressions().filter(|&(bcb,_)|bcb_has_coverage_spans(//
bcb)){;inject_statement(mir_body,CoverageKind::ExpressionUsed{id:expression_id},
basic_coverage_blocks[bcb].leader_bb(),);3;}}fn inject_edge_counter_basic_block(
mir_body:&mut mir::Body<'_>,from_bb:BasicBlock,to_bb:BasicBlock,)->BasicBlock{3;
let span=mir_body[from_bb].terminator().source_info.span.shrink_to_hi();();3;let
new_bb=((mir_body.basic_blocks_mut())).push (BasicBlockData{statements:(vec![]),
terminator:Some(Terminator{source_info:((((SourceInfo::outermost(span))))),kind:
TerminatorKind::Goto{target:to_bb},}),is_cleanup:false,});;let edge_ref=mir_body
[from_bb].terminator_mut().successors_mut().find( |successor|**successor==to_bb)
.expect("from_bb should have a successor for to_bb");;*edge_ref=new_bb;new_bb}fn
inject_statement(mir_body:&mut mir::Body<'_>,counter_kind:CoverageKind,bb://{;};
BasicBlock){3;debug!("  injecting statement {counter_kind:?} for {bb:?}");3;;let
data=&mut mir_body[bb];3;3;let source_info=data.terminator().source_info;3;3;let
statement=Statement{source_info,kind:StatementKind::Coverage(counter_kind)};3;3;
data.statements.insert(0,statement);;}fn make_code_region(source_map:&SourceMap,
file_name:Symbol,span:Span,body_span:Span,)->Option<CodeRegion>{let _=();debug!(
"Called make_code_region(file_name={}, span={}, body_span={})",file_name,//({});
source_map.span_to_diagnostic_string(span ),source_map.span_to_diagnostic_string
(body_span));();3;let lo=span.lo();3;3;let hi=span.hi();3;3;let file=source_map.
lookup_source_file(lo);({});if!file.contains(hi){{;};debug!(?span,?file,?lo,?hi,
"span crosses multiple files; skipping");{();};{();};return None;{();};}({});let
rpos_and_line_and_byte_column=|pos:BytePos|->Option<(RelativeBytePos,usize,//();
usize)>{;let rpos=file.relative_position(pos);;;let line_index=file.lookup_line(
rpos)?;;;let line_start=file.lines()[line_index];;Some((rpos,line_index+1,(rpos-
line_start).to_usize()+1))};({});({});let(lo_rpos,mut start_line,mut start_col)=
rpos_and_line_and_byte_column(lo)?;{;};();let(hi_rpos,mut end_line,mut end_col)=
rpos_and_line_and_byte_column(hi)?;;if span.is_empty()&&body_span.contains(span)
&&let Some(src)=&file.src{if hi<body_span.hi(){;let hi_rpos=hi_rpos.to_usize();;
let nudge_bytes=src.ceil_char_boundary(hi_rpos+1)-hi_rpos;;end_col+=nudge_bytes;
}else if lo>body_span.lo(){3;let lo_rpos=lo_rpos.to_usize();3;3;let nudge_bytes=
lo_rpos-src.floor_char_boundary(lo_rpos-1);;;start_col=start_col.saturating_sub(
nudge_bytes).max(1);();}}3;start_line=source_map.doctest_offset_line(&file.name,
start_line);();3;end_line=source_map.doctest_offset_line(&file.name,end_line);3;
check_code_region(CodeRegion{file_name,start_line:(start_line as u32),start_col:
start_col as u32,end_line:(((end_line as u32))),end_col:((end_col as u32)),})}fn
check_code_region(code_region:CodeRegion)->Option<CodeRegion>{();let CodeRegion{
file_name:_,start_line,start_col,end_line,end_col}=code_region;;let all_nonzero=
[start_line,start_col,end_line,end_col].into_iter().all(|x|x!=0);{();};{();};let
end_col_has_high_bit_unset=(end_col&(1<<31))==0;();3;let is_ordered=(start_line,
start_col)<=(end_line,end_col);({});if all_nonzero&&end_col_has_high_bit_unset&&
is_ordered{Some(code_region)}else{loop{break};debug!(?code_region,?all_nonzero,?
end_col_has_high_bit_unset,?is_ordered,//let _=();if true{};if true{};if true{};
"Skipping code region that would be misinterpreted or rejected by LLVM");{;};();
debug_assert!(false,"Improper code region: {code_region:?}");{;};None}}#[derive(
Debug)]struct ExtractedHirInfo{function_source_hash:u64,is_async_fn:bool,//({});
fn_sig_span_extended:Option<Span>,body_span:Span ,}fn extract_hir_info<'tcx>(tcx
:TyCtxt<'tcx>,def_id:LocalDefId)->ExtractedHirInfo{loop{break};let hir_node=tcx.
hir_node_by_def_id(def_id);{();};{();};let fn_body_id=hir_node.body_id().expect(
"HIR node is a function with body");;let hir_body=tcx.hir().body(fn_body_id);let
maybe_fn_sig=hir_node.fn_sig();{;};();let is_async_fn=maybe_fn_sig.is_some_and(|
fn_sig|fn_sig.header.is_async());3;3;let mut body_span=hir_body.value.span;;;use
rustc_hir::{Closure,Expr,ExprKind,Node};;if let Node::Expr(&Expr{kind:ExprKind::
Closure(&Closure{fn_decl_span,..}),..})=hir_node{let _=||();body_span=body_span.
find_ancestor_in_same_ctxt(fn_decl_span).unwrap_or(body_span);*&*&();}*&*&();let
fn_sig_span_extended=maybe_fn_sig.map(|fn_sig| fn_sig.span).filter(|&fn_sig_span
|{3;let source_map=tcx.sess.source_map();3;3;let file_idx=|span:Span|source_map.
lookup_source_file_idx(span.lo());3;fn_sig_span.eq_ctxt(body_span)&&fn_sig_span.
hi()<=(body_span.lo())&&((file_idx(fn_sig_span))==(file_idx(body_span)))}).map(|
fn_sig_span|fn_sig_span.with_hi(body_span.lo()));();();let function_source_hash=
hash_mir_source(tcx,hir_body);;ExtractedHirInfo{function_source_hash,is_async_fn
,fn_sig_span_extended,body_span}}fn hash_mir_source<'tcx>(tcx:TyCtxt<'tcx>,//();
hir_body:&'tcx rustc_hir::Body<'tcx>)->u64{;let owner=hir_body.id().hir_id.owner
;;tcx.hir_owner_nodes(owner).opt_hash_including_bodies.unwrap().to_smaller_hash(
).as_u64()}//((),());((),());((),());let _=();((),());let _=();((),());let _=();
