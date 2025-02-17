//! Implementation of the LSP for rust-analyzer.
//!
//! This crate takes Rust-specific analysis results from ide and translates
//! into LSP types.
//!
//! It also is the root of all state. `world` module defines the bulk of the
//! state, and `main_loop` module defines the rules for modifying it.
//!
//! The `cli` submodule implements some batch-processing analysis, primarily as
//! a debugging aid.

pub mod cli;

mod command;
mod diagnostics;
mod discover;
mod flycheck;
mod hack_recover_crate_name;
mod line_index;
mod main_loop;
mod mem_docs;
mod op_queue;
mod reload;
mod target_spec;
mod task_pool;
mod test_runner;
mod version;

mod handlers {
    pub(crate) mod dispatch;
    pub(crate) mod notification;
    pub(crate) mod request;
}

pub mod tracing {
    pub mod config;
    pub mod json;
    pub use config::Config;
    pub mod hprof;
}

pub mod config;
mod global_state;
pub mod lsp;
use self::lsp::ext as lsp_ext;

#[cfg(test)]
mod integrated_benchmarks;

use hir::Mutability;
use ide::{CompletionItem, CompletionItemRefMode, CompletionRelevance};
use serde::de::DeserializeOwned;
use tenthash::TentHash;

pub use crate::{
    lsp::capabilities::server_capabilities, main_loop::main_loop, reload::ws_to_crate_graph,
    version::version,
};

pub fn from_json<T: DeserializeOwned>(
    what: &'static str,
    json: &serde_json::Value,
) -> anyhow::Result<T> {
    serde_json::from_value(json.clone())
        .map_err(|e| anyhow::format_err!("Failed to deserialize {what}: {e}; {json}"))
}

fn completion_item_hash(item: &CompletionItem, is_ref_completion: bool) -> [u8; 20] {
    fn hash_completion_relevance(hasher: &mut TentHash, relevance: &CompletionRelevance) {
        use ide_completion::{
            CompletionRelevancePostfixMatch, CompletionRelevanceReturnType,
            CompletionRelevanceTypeMatch,
        };

        hasher.update([
            u8::from(relevance.exact_name_match),
            u8::from(relevance.is_local),
            u8::from(relevance.is_name_already_imported),
            u8::from(relevance.requires_import),
            u8::from(relevance.is_private_editable),
        ]);

        match relevance.type_match {
            None => hasher.update([0u8]),
            Some(CompletionRelevanceTypeMatch::CouldUnify) => hasher.update([1u8]),
            Some(CompletionRelevanceTypeMatch::Exact) => hasher.update([2u8]),
        }

        hasher.update([u8::from(relevance.trait_.is_some())]);
        if let Some(trait_) = &relevance.trait_ {
            hasher.update([u8::from(trait_.is_op_method), u8::from(trait_.notable_trait)]);
        }

        match relevance.postfix_match {
            None => hasher.update([0u8]),
            Some(CompletionRelevancePostfixMatch::NonExact) => hasher.update([1u8]),
            Some(CompletionRelevancePostfixMatch::Exact) => hasher.update([2u8]),
        }

        hasher.update([u8::from(relevance.function.is_some())]);
        if let Some(function) = &relevance.function {
            hasher.update([u8::from(function.has_params), u8::from(function.has_self_param)]);
            let discriminant: u8 = match function.return_type {
                CompletionRelevanceReturnType::Other => 0,
                CompletionRelevanceReturnType::DirectConstructor => 1,
                CompletionRelevanceReturnType::Constructor => 2,
                CompletionRelevanceReturnType::Builder => 3,
            };
            hasher.update([discriminant]);
        }
    }

    let mut hasher = TentHash::new();
    hasher.update([
        u8::from(is_ref_completion),
        u8::from(item.is_snippet),
        u8::from(item.deprecated),
        u8::from(item.trigger_call_info),
    ]);

    hasher.update(item.label.primary.len().to_ne_bytes());
    hasher.update(&item.label.primary);

    hasher.update([u8::from(item.label.detail_left.is_some())]);
    if let Some(label_detail) = &item.label.detail_left {
        hasher.update(label_detail.len().to_ne_bytes());
        hasher.update(label_detail);
    }

    hasher.update([u8::from(item.label.detail_right.is_some())]);
    if let Some(label_detail) = &item.label.detail_right {
        hasher.update(label_detail.len().to_ne_bytes());
        hasher.update(label_detail);
    }

    // NB: do not hash edits or source range, as those may change between the time the client sends the resolve request
    // and the time it receives it: some editors do allow changing the buffer between that, leading to ranges being different.
    //
    // Documentation hashing is skipped too, as it's a large blob to process,
    // while not really making completion properties more unique as they are already.

    let kind_tag = item.kind.tag();
    hasher.update(kind_tag.len().to_ne_bytes());
    hasher.update(kind_tag);

    hasher.update(item.lookup.len().to_ne_bytes());
    hasher.update(&item.lookup);

    hasher.update([u8::from(item.detail.is_some())]);
    if let Some(detail) = &item.detail {
        hasher.update(detail.len().to_ne_bytes());
        hasher.update(detail);
    }

    hash_completion_relevance(&mut hasher, &item.relevance);

    hasher.update([u8::from(item.ref_match.is_some())]);
    if let Some((ref_mode, text_size)) = &item.ref_match {
        let discriminant = match ref_mode {
            CompletionItemRefMode::Reference(Mutability::Shared) => 0u8,
            CompletionItemRefMode::Reference(Mutability::Mut) => 1u8,
            CompletionItemRefMode::Dereference => 2u8,
        };
        hasher.update([discriminant]);
        hasher.update(u32::from(*text_size).to_ne_bytes());
    }

    hasher.update(item.import_to_add.len().to_ne_bytes());
    for import_path in &item.import_to_add {
        hasher.update(import_path.len().to_ne_bytes());
        hasher.update(import_path);
    }

    hasher.finalize()
}
