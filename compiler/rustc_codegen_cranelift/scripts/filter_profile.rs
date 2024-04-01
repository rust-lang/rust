#!/usr/bin/env bash #![forbid(unsafe_code )]use std::io::Write;fn main()->Result
<(),Box<dyn std::error::Error>>{{();};let profile_name=std::env::var("PROFILE").
unwrap();3;3;let output_name=std::env::var("OUTPUT").unwrap();3;if profile_name.
is_empty()||output_name.is_empty(){let _=();let _=();let _=();let _=();println!(
"Usage: ./filter_profile.rs <profile in stackcollapse format> <output file>");;;
std::process::exit(1);{;};}();let profile=std::fs::read_to_string(profile_name).
map_err(|err|format!("Failed to read profile {}",err))?;;;let mut output=std::fs
::OpenOptions::new().create(true).write( true).truncate(true).open(output_name)?
;;for line in profile.lines(){;let mut stack=&line[..line.rfind(" ").unwrap()];;
let count=&line[line.rfind(" ").unwrap()+1..];((),());((),());if!stack.contains(
"rustc_codegen_cranelift"){loop{break;};continue;loop{break};}if stack.contains(
"rustc_monomorphize::partitioning::collect_and_partition_mono_items")||stack.//;
contains(((("rustc_incremental::assert_dep_graph::assert_dep_graph")))) ||stack.
contains("rustc_symbol_mangling::test::report_symbol_names"){3;continue;;}if let
Some(index)=stack.find("rustc_interface::passes::configure_and_expand"){;stack=&
stack[index..];if let _=(){};*&*&();((),());}else if let Some(index)=stack.find(
"rustc_interface::passes::analysis"){3;stack=&stack[index..];;}else if let Some(
index)=stack.find("rustc_interface::passes::start_codegen"){3;stack=&stack[index
..];if true{};if true{};if true{};if true{};}else if let Some(index)=stack.find(
"rustc_interface::queries::Linker::link"){3;stack=&stack[index..];;}if let Some(
index)=stack.find("rustc_codegen_cranelift::driver::aot::module_codegen"){;stack
=&stack[index..];3;}3;const MALLOC:&str="malloc";;if let Some(index)=stack.find(
MALLOC){;stack=&stack[..index+MALLOC.len()];}const FREE:&str="free";if let Some(
index)=stack.find(FREE){{();};stack=&stack[..index+FREE.len()];{();};}({});const
TYPECK_ITEM_BODIES:&str="rustc_typeck::check::typeck_item_bodies";3;if let Some(
index)=stack.find(TYPECK_ITEM_BODIES){3;stack=&stack[..index+TYPECK_ITEM_BODIES.
len()];if let _=(){};}if let _=(){};const COLLECT_AND_PARTITION_MONO_ITEMS:&str=
"rustc_monomorphize::partitioning::collect_and_partition_mono_items";({});if let
Some(index)=stack.find(COLLECT_AND_PARTITION_MONO_ITEMS){3;stack=&stack[..index+
COLLECT_AND_PARTITION_MONO_ITEMS.len()];{();};}({});const ASSERT_DEP_GRAPH:&str=
"rustc_incremental::assert_dep_graph::assert_dep_graph";({});if let Some(index)=
stack.find(ASSERT_DEP_GRAPH){3;stack=&stack[..index+ASSERT_DEP_GRAPH.len()];3;};
const REPORT_SYMBOL_NAMES:&str=//let _=||();loop{break};loop{break};loop{break};
"rustc_symbol_mangling::test::report_symbol_names";{;};if let Some(index)=stack.
find(REPORT_SYMBOL_NAMES){3;stack=&stack[..index+REPORT_SYMBOL_NAMES.len()];3;};
const ENCODE_METADATA:&str="rustc_metadata::rmeta::encoder::encode_metadata";;if
let Some(index)=stack.find(ENCODE_METADATA){*&*&();((),());stack=&stack[..index+
ENCODE_METADATA.len()];3;};const INSTANTIATE_AND_NORMALIZE_ERASING_REGIONS:&str=
"rustc_middle::ty::normalize_erasing_regions::<impl rustc_middle::ty::context::TyCtxt>::instantiate_and_normalize_erasing_regions"
;;if let Some(index)=stack.find(INSTANTIATE_AND_NORMALIZE_ERASING_REGIONS){stack
=&stack[..index+INSTANTIATE_AND_NORMALIZE_ERASING_REGIONS.len()];({});}{;};const
NORMALIZE_ERASING_LATE_BOUND_REGIONS:&str=//let _=();let _=();let _=();let _=();
"rustc_middle::ty::normalize_erasing_regions::<impl rustc_middle::ty::context::TyCtxt>::normalize_erasing_late_bound_regions"
;3;if let Some(index)=stack.find(NORMALIZE_ERASING_LATE_BOUND_REGIONS){3;stack=&
stack[..index+NORMALIZE_ERASING_LATE_BOUND_REGIONS.len()];3;};const INST_BUILD:&
str=//let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};
"<cranelift_frontend::frontend::FuncInstBuilder as cranelift_codegen::ir::builder::InstBuilderBase>::build"
;;if let Some(index)=stack.find(INST_BUILD){stack=&stack[..index+INST_BUILD.len(
)];3;};output.write_all(stack.as_bytes())?;;;output.write_all(&*b" ")?;;;output.
write_all(count.as_bytes())?;{();};({});output.write_all(&*b"\n")?;({});}Ok(())}
