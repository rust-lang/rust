//! Contains information about "passes", used to modify crate information during the documentation
//! process.

use rustc::lint;
use rustc::middle::privacy::AccessLevels;
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_span::{InnerSpan, Span, DUMMY_SP};
use std::mem;
use std::ops::Range;

use self::Condition::*;
use crate::clean::{self, GetDefId, Item};
use crate::core::DocContext;
use crate::fold::{DocFolder, StripItem};
use crate::html::markdown::{find_testable_code, ErrorCodes, LangString};

mod collapse_docs;
pub use self::collapse_docs::COLLAPSE_DOCS;

mod strip_hidden;
pub use self::strip_hidden::STRIP_HIDDEN;

mod strip_private;
pub use self::strip_private::STRIP_PRIVATE;

mod strip_priv_imports;
pub use self::strip_priv_imports::STRIP_PRIV_IMPORTS;

mod unindent_comments;
pub use self::unindent_comments::UNINDENT_COMMENTS;

mod propagate_doc_cfg;
pub use self::propagate_doc_cfg::PROPAGATE_DOC_CFG;

mod collect_intra_doc_links;
pub use self::collect_intra_doc_links::COLLECT_INTRA_DOC_LINKS;

mod private_items_doc_tests;
pub use self::private_items_doc_tests::CHECK_PRIVATE_ITEMS_DOC_TESTS;

mod collect_trait_impls;
pub use self::collect_trait_impls::COLLECT_TRAIT_IMPLS;

mod check_code_block_syntax;
pub use self::check_code_block_syntax::CHECK_CODE_BLOCK_SYNTAX;

mod calculate_doc_coverage;
pub use self::calculate_doc_coverage::CALCULATE_DOC_COVERAGE;

/// A single pass over the cleaned documentation.
///
/// Runs in the compiler context, so it has access to types and traits and the like.
#[derive(Copy, Clone)]
pub struct Pass {
    pub name: &'static str,
    pub run: fn(clean::Crate, &DocContext<'_>) -> clean::Crate,
    pub description: &'static str,
}

/// In a list of passes, a pass that may or may not need to be run depending on options.
#[derive(Copy, Clone)]
pub struct ConditionalPass {
    pub pass: Pass,
    pub condition: Condition,
}

/// How to decide whether to run a conditional pass.
#[derive(Copy, Clone)]
pub enum Condition {
    Always,
    /// When `--document-private-items` is passed.
    WhenDocumentPrivate,
    /// When `--document-private-items` is not passed.
    WhenNotDocumentPrivate,
    /// When `--document-hidden-items` is not passed.
    WhenNotDocumentHidden,
}

/// The full list of passes.
pub const PASSES: &[Pass] = &[
    CHECK_PRIVATE_ITEMS_DOC_TESTS,
    STRIP_HIDDEN,
    UNINDENT_COMMENTS,
    COLLAPSE_DOCS,
    STRIP_PRIVATE,
    STRIP_PRIV_IMPORTS,
    PROPAGATE_DOC_CFG,
    COLLECT_INTRA_DOC_LINKS,
    CHECK_CODE_BLOCK_SYNTAX,
    COLLECT_TRAIT_IMPLS,
    CALCULATE_DOC_COVERAGE,
];

/// The list of passes run by default.
pub const DEFAULT_PASSES: &[ConditionalPass] = &[
    ConditionalPass::always(COLLECT_TRAIT_IMPLS),
    ConditionalPass::always(COLLAPSE_DOCS),
    ConditionalPass::always(UNINDENT_COMMENTS),
    ConditionalPass::always(CHECK_PRIVATE_ITEMS_DOC_TESTS),
    ConditionalPass::new(STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::new(STRIP_PRIV_IMPORTS, WhenDocumentPrivate),
    ConditionalPass::always(COLLECT_INTRA_DOC_LINKS),
    ConditionalPass::always(CHECK_CODE_BLOCK_SYNTAX),
    ConditionalPass::always(PROPAGATE_DOC_CFG),
];

/// The list of default passes run when `--doc-coverage` is passed to rustdoc.
pub const COVERAGE_PASSES: &[ConditionalPass] = &[
    ConditionalPass::always(COLLECT_TRAIT_IMPLS),
    ConditionalPass::new(STRIP_HIDDEN, WhenNotDocumentHidden),
    ConditionalPass::new(STRIP_PRIVATE, WhenNotDocumentPrivate),
    ConditionalPass::always(CALCULATE_DOC_COVERAGE),
];

impl ConditionalPass {
    pub const fn always(pass: Pass) -> Self {
        Self::new(pass, Always)
    }

    pub const fn new(pass: Pass, condition: Condition) -> Self {
        ConditionalPass { pass, condition }
    }
}

/// A shorthand way to refer to which set of passes to use, based on the presence of
/// `--no-defaults` and `--show-coverage`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DefaultPassOption {
    Default,
    Coverage,
    None,
}

/// Returns the given default set of passes.
pub fn defaults(default_set: DefaultPassOption) -> &'static [ConditionalPass] {
    match default_set {
        DefaultPassOption::Default => DEFAULT_PASSES,
        DefaultPassOption::Coverage => COVERAGE_PASSES,
        DefaultPassOption::None => &[],
    }
}

/// If the given name matches a known pass, returns its information.
pub fn find_pass(pass_name: &str) -> Option<Pass> {
    PASSES.iter().find(|p| p.name == pass_name).copied()
}

struct Stripper<'a> {
    retained: &'a mut DefIdSet,
    access_levels: &'a AccessLevels<DefId>,
    update_retained: bool,
}

impl<'a> DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::StrippedItem(..) => {
                // We need to recurse into stripped modules to strip things
                // like impl methods but when doing so we must not add any
                // items to the `retained` set.
                debug!("Stripper: recursing into stripped {:?} {:?}", i.type_(), i.name);
                let old = mem::replace(&mut self.update_retained, false);
                let ret = self.fold_item_recur(i);
                self.update_retained = old;
                return ret;
            }
            // These items can all get re-exported
            clean::OpaqueTyItem(..)
            | clean::TypedefItem(..)
            | clean::StaticItem(..)
            | clean::StructItem(..)
            | clean::EnumItem(..)
            | clean::TraitItem(..)
            | clean::FunctionItem(..)
            | clean::VariantItem(..)
            | clean::MethodItem(..)
            | clean::ForeignFunctionItem(..)
            | clean::ForeignStaticItem(..)
            | clean::ConstantItem(..)
            | clean::UnionItem(..)
            | clean::AssocConstItem(..)
            | clean::TraitAliasItem(..)
            | clean::ForeignTypeItem => {
                if i.def_id.is_local() {
                    if !self.access_levels.is_exported(i.def_id) {
                        debug!("Stripper: stripping {:?} {:?}", i.type_(), i.name);
                        return None;
                    }
                }
            }

            clean::StructFieldItem(..) => {
                if i.visibility != clean::Public {
                    return StripItem(i).strip();
                }
            }

            clean::ModuleItem(..) => {
                if i.def_id.is_local() && i.visibility != clean::Public {
                    debug!("Stripper: stripping module {:?}", i.name);
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = StripItem(self.fold_item_recur(i).unwrap()).strip();
                    self.update_retained = old;
                    return ret;
                }
            }

            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem(..) | clean::ImportItem(..) => {}

            clean::ImplItem(..) => {}

            // tymethods/macros have no control over privacy
            clean::MacroItem(..) | clean::TyMethodItem(..) => {}

            // Proc-macros are always public
            clean::ProcMacroItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Associated types are never stripped
            clean::AssocTypeItem(..) => {}

            // Keywords are never stripped
            clean::KeywordItem(..) => {}
        }

        let fastreturn = match i.inner {
            // nothing left to do for traits (don't want to filter their
            // methods out, visibility controlled by the trait)
            clean::TraitItem(..) => true,

            // implementations of traits are always public.
            clean::ImplItem(ref imp) if imp.trait_.is_some() => true,
            // Struct variant fields have inherited visibility
            clean::VariantItem(clean::Variant { kind: clean::VariantKind::Struct(..) }) => true,
            _ => false,
        };

        let i = if fastreturn {
            if self.update_retained {
                self.retained.insert(i.def_id);
            }
            return Some(i);
        } else {
            self.fold_item_recur(i)
        };

        if let Some(ref i) = i {
            if self.update_retained {
                self.retained.insert(i.def_id);
            }
        }
        i
    }
}

// This stripper discards all impls which reference stripped items
struct ImplStripper<'a> {
    retained: &'a DefIdSet,
}

impl<'a> DocFolder for ImplStripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = i.inner {
            // emptied none trait impls can be stripped
            if imp.trait_.is_none() && imp.items.is_empty() {
                return None;
            }
            if let Some(did) = imp.for_.def_id() {
                if did.is_local() && !imp.for_.is_generic() && !self.retained.contains(&did) {
                    debug!("ImplStripper: impl item for stripped type; removing");
                    return None;
                }
            }
            if let Some(did) = imp.trait_.def_id() {
                if did.is_local() && !self.retained.contains(&did) {
                    debug!("ImplStripper: impl item for stripped trait; removing");
                    return None;
                }
            }
            if let Some(generics) = imp.trait_.as_ref().and_then(|t| t.generics()) {
                for typaram in generics {
                    if let Some(did) = typaram.def_id() {
                        if did.is_local() && !self.retained.contains(&did) {
                            debug!(
                                "ImplStripper: stripped item in trait's generics; \
                                    removing impl"
                            );
                            return None;
                        }
                    }
                }
            }
        }
        self.fold_item_recur(i)
    }
}

// This stripper discards all private import statements (`use`, `extern crate`)
struct ImportStripper;
impl DocFolder for ImportStripper {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::ExternCrateItem(..) | clean::ImportItem(..) if i.visibility != clean::Public => {
                None
            }
            _ => self.fold_item_recur(i),
        }
    }
}

pub fn look_for_tests<'tcx>(
    cx: &DocContext<'tcx>,
    dox: &str,
    item: &Item,
    check_missing_code: bool,
) {
    let hir_id = match cx.as_local_hir_id(item.def_id) {
        Some(hir_id) => hir_id,
        None => {
            // If non-local, no need to check anything.
            return;
        }
    };

    struct Tests {
        found_tests: usize,
    }

    impl crate::test::Tester for Tests {
        fn add_test(&mut self, _: String, _: LangString, _: usize) {
            self.found_tests += 1;
        }
    }

    let mut tests = Tests { found_tests: 0 };

    find_testable_code(&dox, &mut tests, ErrorCodes::No, false);

    if check_missing_code == true && tests.found_tests == 0 {
        let sp = span_of_attrs(&item.attrs).unwrap_or(item.source.span());
        cx.tcx.struct_span_lint_hir(lint::builtin::MISSING_DOC_CODE_EXAMPLES, hir_id, sp, |lint| {
            lint.build("missing code example in this documentation").emit()
        });
    } else if check_missing_code == false
        && tests.found_tests > 0
        && !cx.renderinfo.borrow().access_levels.is_public(item.def_id)
    {
        cx.tcx.struct_span_lint_hir(
            lint::builtin::PRIVATE_DOC_TESTS,
            hir_id,
            span_of_attrs(&item.attrs).unwrap_or(item.source.span()),
            |lint| lint.build("documentation test in private item").emit(),
        );
    }
}

/// Returns a span encompassing all the given attributes.
crate fn span_of_attrs(attrs: &clean::Attributes) -> Option<Span> {
    if attrs.doc_strings.is_empty() {
        return None;
    }
    let start = attrs.doc_strings[0].span();
    if start == DUMMY_SP {
        return None;
    }
    let end = attrs.doc_strings.last().expect("no doc strings provided").span();
    Some(start.to(end))
}

/// Attempts to match a range of bytes from parsed markdown to a `Span` in the source code.
///
/// This method will return `None` if we cannot construct a span from the source map or if the
/// attributes are not all sugared doc comments. It's difficult to calculate the correct span in
/// that case due to escaping and other source features.
crate fn source_span_for_markdown_range(
    cx: &DocContext<'_>,
    markdown: &str,
    md_range: &Range<usize>,
    attrs: &clean::Attributes,
) -> Option<Span> {
    let is_all_sugared_doc = attrs.doc_strings.iter().all(|frag| match frag {
        clean::DocFragment::SugaredDoc(..) => true,
        _ => false,
    });

    if !is_all_sugared_doc {
        return None;
    }

    let snippet = cx.sess().source_map().span_to_snippet(span_of_attrs(attrs)?).ok()?;

    let starting_line = markdown[..md_range.start].matches('\n').count();
    let ending_line = starting_line + markdown[md_range.start..md_range.end].matches('\n').count();

    // We use `split_terminator('\n')` instead of `lines()` when counting bytes so that we treat
    // CRLF and LF line endings the same way.
    let mut src_lines = snippet.split_terminator('\n');
    let md_lines = markdown.split_terminator('\n');

    // The number of bytes from the source span to the markdown span that are not part
    // of the markdown, like comment markers.
    let mut start_bytes = 0;
    let mut end_bytes = 0;

    'outer: for (line_no, md_line) in md_lines.enumerate() {
        loop {
            let source_line = src_lines.next().expect("could not find markdown in source");
            match source_line.find(md_line) {
                Some(offset) => {
                    if line_no == starting_line {
                        start_bytes += offset;

                        if starting_line == ending_line {
                            break 'outer;
                        }
                    } else if line_no == ending_line {
                        end_bytes += offset;
                        break 'outer;
                    } else if line_no < starting_line {
                        start_bytes += source_line.len() - md_line.len();
                    } else {
                        end_bytes += source_line.len() - md_line.len();
                    }
                    break;
                }
                None => {
                    // Since this is a source line that doesn't include a markdown line,
                    // we have to count the newline that we split from earlier.
                    if line_no <= starting_line {
                        start_bytes += source_line.len() + 1;
                    } else {
                        end_bytes += source_line.len() + 1;
                    }
                }
            }
        }
    }

    Some(span_of_attrs(attrs)?.from_inner(InnerSpan::new(
        md_range.start + start_bytes,
        md_range.end + start_bytes + end_bytes,
    )))
}
