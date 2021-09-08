#!/bin/bash
#![forbid(unsafe_code)]/* This line is ignored by bash
# This block is ignored by rustc
pushd $(dirname "$0")/../
source scripts/config.sh
RUSTC="$(pwd)/build/bin/cg_clif"
popd
PROFILE=$1 OUTPUT=$2 exec $RUSTC -Zunstable-options -Cllvm-args=mode=jit -Cprefer-dynamic $0
#*/

//! This program filters away uninteresting samples and trims uninteresting frames for stackcollapse
//! profiles.
//!
//! Usage: ./filter_profile.rs <profile in stackcollapse format> <output file>
//!
//! This file is specially crafted to be both a valid bash script and valid rust source file. If
//! executed as bash script this will run the rust source using cg_clif in JIT mode.

use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile_name = std::env::var("PROFILE").unwrap();
    let output_name = std::env::var("OUTPUT").unwrap();
    if profile_name.is_empty() || output_name.is_empty() {
        println!("Usage: ./filter_profile.rs <profile in stackcollapse format> <output file>");
        std::process::exit(1);
    }
    let profile = std::fs::read_to_string(profile_name)
        .map_err(|err| format!("Failed to read profile {}", err))?;
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_name)?;

    for line in profile.lines() {
        let mut stack = &line[..line.rfind(" ").unwrap()];
        let count = &line[line.rfind(" ").unwrap() + 1..];

        // Filter away uninteresting samples
        if !stack.contains("rustc_codegen_cranelift") {
            continue;
        }

        if stack.contains("rustc_monomorphize::partitioning::collect_and_partition_mono_items")
            || stack.contains("rustc_incremental::assert_dep_graph::assert_dep_graph")
            || stack.contains("rustc_symbol_mangling::test::report_symbol_names")
        {
            continue;
        }

        // Trim start
        if let Some(index) = stack.find("rustc_interface::passes::configure_and_expand") {
            stack = &stack[index..];
        } else if let Some(index) = stack.find("rustc_interface::passes::analysis") {
            stack = &stack[index..];
        } else if let Some(index) = stack.find("rustc_interface::passes::start_codegen") {
            stack = &stack[index..];
        } else if let Some(index) = stack.find("rustc_interface::queries::Linker::link") {
            stack = &stack[index..];
        }

        if let Some(index) = stack.find("rustc_codegen_cranelift::driver::aot::module_codegen") {
            stack = &stack[index..];
        }

        // Trim end
        const MALLOC: &str = "malloc";
        if let Some(index) = stack.find(MALLOC) {
            stack = &stack[..index + MALLOC.len()];
        }

        const FREE: &str = "free";
        if let Some(index) = stack.find(FREE) {
            stack = &stack[..index + FREE.len()];
        }

        const TYPECK_ITEM_BODIES: &str = "rustc_typeck::check::typeck_item_bodies";
        if let Some(index) = stack.find(TYPECK_ITEM_BODIES) {
            stack = &stack[..index + TYPECK_ITEM_BODIES.len()];
        }

        const COLLECT_AND_PARTITION_MONO_ITEMS: &str =
            "rustc_monomorphize::partitioning::collect_and_partition_mono_items";
        if let Some(index) = stack.find(COLLECT_AND_PARTITION_MONO_ITEMS) {
            stack = &stack[..index + COLLECT_AND_PARTITION_MONO_ITEMS.len()];
        }

        const ASSERT_DEP_GRAPH: &str = "rustc_incremental::assert_dep_graph::assert_dep_graph";
        if let Some(index) = stack.find(ASSERT_DEP_GRAPH) {
            stack = &stack[..index + ASSERT_DEP_GRAPH.len()];
        }

        const REPORT_SYMBOL_NAMES: &str = "rustc_symbol_mangling::test::report_symbol_names";
        if let Some(index) = stack.find(REPORT_SYMBOL_NAMES) {
            stack = &stack[..index + REPORT_SYMBOL_NAMES.len()];
        }

        const ENCODE_METADATA: &str = "rustc_middle::ty::context::TyCtxt::encode_metadata";
        if let Some(index) = stack.find(ENCODE_METADATA) {
            stack = &stack[..index + ENCODE_METADATA.len()];
        }

        const SUBST_AND_NORMALIZE_ERASING_REGIONS: &str = "rustc_middle::ty::normalize_erasing_regions::<impl rustc_middle::ty::context::TyCtxt>::subst_and_normalize_erasing_regions";
        if let Some(index) = stack.find(SUBST_AND_NORMALIZE_ERASING_REGIONS) {
            stack = &stack[..index + SUBST_AND_NORMALIZE_ERASING_REGIONS.len()];
        }

        const NORMALIZE_ERASING_LATE_BOUND_REGIONS: &str = "rustc_middle::ty::normalize_erasing_regions::<impl rustc_middle::ty::context::TyCtxt>::normalize_erasing_late_bound_regions";
        if let Some(index) = stack.find(NORMALIZE_ERASING_LATE_BOUND_REGIONS) {
            stack = &stack[..index + NORMALIZE_ERASING_LATE_BOUND_REGIONS.len()];
        }

        const INST_BUILD: &str = "<cranelift_frontend::frontend::FuncInstBuilder as cranelift_codegen::ir::builder::InstBuilderBase>::build";
        if let Some(index) = stack.find(INST_BUILD) {
            stack = &stack[..index + INST_BUILD.len()];
        }

        output.write_all(stack.as_bytes())?;
        output.write_all(&*b" ")?;
        output.write_all(count.as_bytes())?;
        output.write_all(&*b"\n")?;
    }

    Ok(())
}
