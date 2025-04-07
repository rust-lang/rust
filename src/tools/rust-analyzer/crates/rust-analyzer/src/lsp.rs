//! Custom LSP definitions and protocol conversions.

use core::fmt;

use hir::Mutability;
use ide::{CompletionItem, CompletionItemRefMode, CompletionRelevance};
use tenthash::TentHash;

pub mod ext;

pub(crate) mod capabilities;
pub(crate) mod from_proto;
pub(crate) mod semantic_tokens;
pub(crate) mod to_proto;
pub(crate) mod utils;

#[derive(Debug)]
pub(crate) struct LspError {
    pub(crate) code: i32,
    pub(crate) message: String,
}

impl LspError {
    pub(crate) fn new(code: i32, message: String) -> LspError {
        LspError { code, message }
    }
}

impl fmt::Display for LspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Language Server request failed with {}. ({})", self.code, self.message)
    }
}

impl std::error::Error for LspError {}

pub(crate) fn completion_item_hash(item: &CompletionItem, is_ref_completion: bool) -> [u8; 20] {
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
