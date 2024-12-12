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

use ide::{CompletionItem, CompletionRelevance};
use serde::de::DeserializeOwned;
use tenthash::TentHasher;

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
    fn hash_completion_relevance(hasher: &mut TentHasher, relevance: &CompletionRelevance) {
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
        if let Some(type_match) = &relevance.type_match {
            let label = match type_match {
                CompletionRelevanceTypeMatch::CouldUnify => "could_unify",
                CompletionRelevanceTypeMatch::Exact => "exact",
            };
            hasher.update(label);
        }
        if let Some(trait_) = &relevance.trait_ {
            hasher.update([u8::from(trait_.is_op_method), u8::from(trait_.notable_trait)]);
        }
        if let Some(postfix_match) = &relevance.postfix_match {
            let label = match postfix_match {
                CompletionRelevancePostfixMatch::NonExact => "non_exact",
                CompletionRelevancePostfixMatch::Exact => "exact",
            };
            hasher.update(label);
        }
        if let Some(function) = &relevance.function {
            hasher.update([u8::from(function.has_params), u8::from(function.has_self_param)]);
            let label = match function.return_type {
                CompletionRelevanceReturnType::Other => "other",
                CompletionRelevanceReturnType::DirectConstructor => "direct_constructor",
                CompletionRelevanceReturnType::Constructor => "constructor",
                CompletionRelevanceReturnType::Builder => "builder",
            };
            hasher.update(label);
        }
    }

    let mut hasher = TentHasher::new();
    hasher.update([
        u8::from(is_ref_completion),
        u8::from(item.is_snippet),
        u8::from(item.deprecated),
        u8::from(item.trigger_call_info),
    ]);
    hasher.update(&item.label);
    if let Some(label_detail) = &item.label_detail {
        hasher.update(label_detail);
    }
    // NB: do not hash edits or source range, as those may change between the time the client sends the resolve request
    // and the time it receives it: some editors do allow changing the buffer between that, leading to ranges being different.
    //
    // Documentation hashing is skipped too, as it's a large blob to process,
    // while not really making completion properties more unique as they are already.
    hasher.update(item.kind.tag());
    hasher.update(&item.lookup);
    if let Some(detail) = &item.detail {
        hasher.update(detail);
    }
    hash_completion_relevance(&mut hasher, &item.relevance);
    if let Some((mutability, text_size)) = &item.ref_match {
        hasher.update(mutability.as_keyword_for_ref());
        hasher.update(u32::from(*text_size).to_le_bytes());
    }
    for (import_path, import_name) in &item.import_to_add {
        hasher.update(import_path);
        hasher.update(import_name);
    }
    hasher.finalize()
}
