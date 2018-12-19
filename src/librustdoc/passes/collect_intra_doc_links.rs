// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::lint as lint;
use rustc::hir;
use rustc::hir::def::Def;
use rustc::ty;
use syntax;
use syntax::ast::{self, Ident, NodeId};
use syntax::feature_gate::UnstableFeatures;
use syntax::symbol::Symbol;
use syntax_pos::{self, DUMMY_SP};

use std::ops::Range;

use core::DocContext;
use fold::DocFolder;
use html::markdown::markdown_links;

use clean::*;
use passes::{look_for_tests, Pass};

pub const COLLECT_INTRA_DOC_LINKS: Pass =
    Pass::early("collect-intra-doc-links", collect_intra_doc_links,
                "reads a crate's documentation to resolve intra-doc-links");

pub fn collect_intra_doc_links(krate: Crate, cx: &DocContext) -> Crate {
    if !UnstableFeatures::from_environment().is_nightly_build() {
        krate
    } else {
        let mut coll = LinkCollector::new(cx);

        coll.fold_crate(krate)
    }
}

#[derive(Debug)]
enum PathKind {
    /// Either a value or type, but not a macro
    Unknown,
    /// Macro
    Macro,
    /// Values, functions, consts, statics (everything in the value namespace)
    Value,
    /// Types, traits (everything in the type namespace)
    Type,
}

struct LinkCollector<'a, 'tcx: 'a, 'rcx: 'a> {
    cx: &'a DocContext<'a, 'tcx, 'rcx>,
    mod_ids: Vec<NodeId>,
    is_nightly_build: bool,
}

impl<'a, 'tcx, 'rcx> LinkCollector<'a, 'tcx, 'rcx> {
    fn new(cx: &'a DocContext<'a, 'tcx, 'rcx>) -> Self {
        LinkCollector {
            cx,
            mod_ids: Vec::new(),
            is_nightly_build: UnstableFeatures::from_environment().is_nightly_build(),
        }
    }

    /// Resolve a given string as a path, along with whether or not it is
    /// in the value namespace. Also returns an optional URL fragment in the case
    /// of variants and methods.
    fn resolve(&self,
               path_str: &str,
               is_val: bool,
               current_item: &Option<String>,
               parent_id: Option<NodeId>)
        -> Result<(Def, Option<String>), ()>
    {
        let cx = self.cx;

        // In case we're in a module, try to resolve the relative
        // path.
        if let Some(id) = parent_id.or(self.mod_ids.last().cloned()) {
            // FIXME: `with_scope` requires the `NodeId` of a module.
            let result = cx.resolver.borrow_mut()
                                    .with_scope(id,
                |resolver| {
                    resolver.resolve_str_path_error(DUMMY_SP,
                                                    &path_str, is_val)
            });

            if let Ok(result) = result {
                // In case this is a trait item, skip the
                // early return and try looking for the trait.
                let value = match result.def {
                    Def::Method(_) | Def::AssociatedConst(_) => true,
                    Def::AssociatedTy(_) => false,
                    Def::Variant(_) => return handle_variant(cx, result.def),
                    // Not a trait item; just return what we found.
                    _ => return Ok((result.def, None))
                };

                if value != is_val {
                    return Err(())
                }
            } else if let Some(prim) = is_primitive(path_str, is_val) {
                return Ok((prim, Some(path_str.to_owned())))
            } else {
                // If resolution failed, it may still be a method
                // because methods are not handled by the resolver
                // If so, bail when we're not looking for a value.
                if !is_val {
                    return Err(())
                }
            }

            // Try looking for methods and associated items.
            let mut split = path_str.rsplitn(2, "::");
            let item_name = if let Some(first) = split.next() {
                first
            } else {
                return Err(())
            };

            let mut path = if let Some(second) = split.next() {
                second.to_owned()
            } else {
                return Err(())
            };

            if path == "self" || path == "Self" {
                if let Some(name) = current_item.as_ref() {
                    path = name.clone();
                }
            }

            // FIXME: `with_scope` requires the `NodeId` of a module.
            let ty = cx.resolver.borrow_mut()
                                .with_scope(id,
                |resolver| {
                    resolver.resolve_str_path_error(DUMMY_SP, &path, false)
            })?;
            match ty.def {
                Def::Struct(did) | Def::Union(did) | Def::Enum(did) | Def::TyAlias(did) => {
                    let item = cx.tcx.inherent_impls(did)
                                     .iter()
                                     .flat_map(|imp| cx.tcx.associated_items(*imp))
                                     .find(|item| item.ident.name == item_name);
                    if let Some(item) = item {
                        let out = match item.kind {
                            ty::AssociatedKind::Method if is_val => "method",
                            ty::AssociatedKind::Const if is_val => "associatedconstant",
                            _ => return Err(())
                        };
                        Ok((ty.def, Some(format!("{}.{}", out, item_name))))
                    } else {
                        match cx.tcx.type_of(did).sty {
                            ty::Adt(def, _) => {
                                if let Some(item) = if def.is_enum() {
                                    def.all_fields().find(|item| item.ident.name == item_name)
                                } else {
                                    def.non_enum_variant()
                                       .fields
                                       .iter()
                                       .find(|item| item.ident.name == item_name)
                                } {
                                    Ok((ty.def,
                                        Some(format!("{}.{}",
                                                     if def.is_enum() {
                                                         "variant"
                                                     } else {
                                                         "structfield"
                                                     },
                                                     item.ident))))
                                } else {
                                    Err(())
                                }
                            }
                            _ => Err(()),
                        }
                    }
                }
                Def::Trait(did) => {
                    let item = cx.tcx.associated_item_def_ids(did).iter()
                                 .map(|item| cx.tcx.associated_item(*item))
                                 .find(|item| item.ident.name == item_name);
                    if let Some(item) = item {
                        let kind = match item.kind {
                            ty::AssociatedKind::Const if is_val => "associatedconstant",
                            ty::AssociatedKind::Type if !is_val => "associatedtype",
                            ty::AssociatedKind::Method if is_val => {
                                if item.defaultness.has_value() {
                                    "method"
                                } else {
                                    "tymethod"
                                }
                            }
                            _ => return Err(())
                        };

                        Ok((ty.def, Some(format!("{}.{}", kind, item_name))))
                    } else {
                        Err(())
                    }
                }
                _ => Err(())
            }
        } else {
            Err(())
        }
    }
}

impl<'a, 'tcx, 'rcx> DocFolder for LinkCollector<'a, 'tcx, 'rcx> {
    fn fold_item(&mut self, mut item: Item) -> Option<Item> {
        let item_node_id = if item.is_mod() {
            if let Some(id) = self.cx.tcx.hir().as_local_node_id(item.def_id) {
                Some(id)
            } else {
                debug!("attempting to fold on a non-local item: {:?}", item);
                return self.fold_item_recur(item);
            }
        } else {
            None
        };

        // FIXME: get the resolver to work with non-local resolve scopes.
        let parent_node = self.cx.as_local_node_id(item.def_id).and_then(|node_id| {
            // FIXME: this fails hard for impls in non-module scope, but is necessary for the
            // current `resolve()` implementation.
            match self.cx.tcx.hir().get_module_parent_node(node_id) {
                id if id != node_id => Some(id),
                _ => None,
            }
        });

        if parent_node.is_some() {
            debug!("got parent node for {} {:?}, id {:?}", item.type_(), item.name, item.def_id);
        }

        let current_item = match item.inner {
            ModuleItem(..) => {
                if item.attrs.inner_docs {
                    if item_node_id.unwrap() != NodeId::from_u32(0) {
                        item.name.clone()
                    } else {
                        None
                    }
                } else {
                    match parent_node.or(self.mod_ids.last().cloned()) {
                        Some(parent) if parent != NodeId::from_u32(0) => {
                            // FIXME: can we pull the parent module's name from elsewhere?
                            Some(self.cx.tcx.hir().name(parent).to_string())
                        }
                        _ => None,
                    }
                }
            }
            ImplItem(Impl { ref for_, .. }) => {
                for_.def_id().map(|did| self.cx.tcx.item_name(did).to_string())
            }
            // we don't display docs on `extern crate` items anyway, so don't process them.
            ExternCrateItem(..) => return self.fold_item_recur(item),
            ImportItem(Import::Simple(ref name, ..)) => Some(name.clone()),
            MacroItem(..) => None,
            _ => item.name.clone(),
        };

        if item.is_mod() && item.attrs.inner_docs {
            self.mod_ids.push(item_node_id.unwrap());
        }

        let cx = self.cx;
        let dox = item.attrs.collapsed_doc_value().unwrap_or_else(String::new);

        look_for_tests(&cx, &dox, &item, true);

        if !self.is_nightly_build {
            return None;
        }

        for (ori_link, link_range) in markdown_links(&dox) {
            // Bail early for real links.
            if ori_link.contains('/') {
                continue;
            }
            let link = ori_link.replace("`", "");
            let (def, fragment) = {
                let mut kind = PathKind::Unknown;
                let path_str = if let Some(prefix) =
                    ["struct@", "enum@", "type@",
                     "trait@", "union@"].iter()
                                      .find(|p| link.starts_with(**p)) {
                    kind = PathKind::Type;
                    link.trim_start_matches(prefix)
                } else if let Some(prefix) =
                    ["const@", "static@",
                     "value@", "function@", "mod@",
                     "fn@", "module@", "method@"]
                        .iter().find(|p| link.starts_with(**p)) {
                    kind = PathKind::Value;
                    link.trim_start_matches(prefix)
                } else if link.ends_with("()") {
                    kind = PathKind::Value;
                    link.trim_end_matches("()")
                } else if link.starts_with("macro@") {
                    kind = PathKind::Macro;
                    link.trim_start_matches("macro@")
                } else if link.ends_with('!') {
                    kind = PathKind::Macro;
                    link.trim_end_matches('!')
                } else {
                    &link[..]
                }.trim();

                if path_str.contains(|ch: char| !(ch.is_alphanumeric() ||
                                                  ch == ':' || ch == '_')) {
                    continue;
                }

                match kind {
                    PathKind::Value => {
                        if let Ok(def) = self.resolve(path_str, true, &current_item, parent_node) {
                            def
                        } else {
                            resolution_failure(cx, &item.attrs, path_str, &dox, link_range);
                            // This could just be a normal link or a broken link
                            // we could potentially check if something is
                            // "intra-doc-link-like" and warn in that case.
                            continue;
                        }
                    }
                    PathKind::Type => {
                        if let Ok(def) = self.resolve(path_str, false, &current_item, parent_node) {
                            def
                        } else {
                            resolution_failure(cx, &item.attrs, path_str, &dox, link_range);
                            // This could just be a normal link.
                            continue;
                        }
                    }
                    PathKind::Unknown => {
                        // Try everything!
                        if let Some(macro_def) = macro_resolve(cx, path_str) {
                            if let Ok(type_def) =
                                self.resolve(path_str, false, &current_item, parent_node)
                            {
                                let (type_kind, article, type_disambig)
                                    = type_ns_kind(type_def.0, path_str);
                                ambiguity_error(cx, &item.attrs, path_str,
                                                article, type_kind, &type_disambig,
                                                "a", "macro", &format!("macro@{}", path_str));
                                continue;
                            } else if let Ok(value_def) =
                                self.resolve(path_str, true, &current_item, parent_node)
                            {
                                let (value_kind, value_disambig)
                                    = value_ns_kind(value_def.0, path_str)
                                        .expect("struct and mod cases should have been \
                                                 caught in previous branch");
                                ambiguity_error(cx, &item.attrs, path_str,
                                                "a", value_kind, &value_disambig,
                                                "a", "macro", &format!("macro@{}", path_str));
                            }
                            (macro_def, None)
                        } else if let Ok(type_def) =
                            self.resolve(path_str, false, &current_item, parent_node)
                        {
                            // It is imperative we search for not-a-value first
                            // Otherwise we will find struct ctors for when we are looking
                            // for structs, and the link won't work if there is something in
                            // both namespaces.
                            if let Ok(value_def) =
                                self.resolve(path_str, true, &current_item, parent_node)
                            {
                                let kind = value_ns_kind(value_def.0, path_str);
                                if let Some((value_kind, value_disambig)) = kind {
                                    let (type_kind, article, type_disambig)
                                        = type_ns_kind(type_def.0, path_str);
                                    ambiguity_error(cx, &item.attrs, path_str,
                                                    article, type_kind, &type_disambig,
                                                    "a", value_kind, &value_disambig);
                                    continue;
                                }
                            }
                            type_def
                        } else if let Ok(value_def) =
                            self.resolve(path_str, true, &current_item, parent_node)
                        {
                            value_def
                        } else {
                            resolution_failure(cx, &item.attrs, path_str, &dox, link_range);
                            // this could just be a normal link
                            continue;
                        }
                    }
                    PathKind::Macro => {
                        if let Some(def) = macro_resolve(cx, path_str) {
                            (def, None)
                        } else {
                            resolution_failure(cx, &item.attrs, path_str, &dox, link_range);
                            continue
                        }
                    }
                }
            };

            if let Def::PrimTy(_) = def {
                item.attrs.links.push((ori_link, None, fragment));
            } else {
                let id = register_def(cx, def);
                item.attrs.links.push((ori_link, Some(id), fragment));
            }
        }

        if item.is_mod() && !item.attrs.inner_docs {
            self.mod_ids.push(item_node_id.unwrap());
        }

        if item.is_mod() {
            let ret = self.fold_item_recur(item);

            self.mod_ids.pop();

            ret
        } else {
            self.fold_item_recur(item)
        }
    }
}

/// Resolve a string as a macro.
fn macro_resolve(cx: &DocContext, path_str: &str) -> Option<Def> {
    use syntax::ext::base::{MacroKind, SyntaxExtension};
    let segment = ast::PathSegment::from_ident(Ident::from_str(path_str));
    let path = ast::Path { segments: vec![segment], span: DUMMY_SP };
    let mut resolver = cx.resolver.borrow_mut();
    let parent_scope = resolver.dummy_parent_scope();
    if let Ok(def) = resolver.resolve_macro_to_def_inner(&path, MacroKind::Bang,
                                                         &parent_scope, false, false) {
        if let SyntaxExtension::DeclMacro { .. } = *resolver.get_macro(def) {
            return Some(def);
        }
    }
    if let Some(def) = resolver.all_macros.get(&Symbol::intern(path_str)) {
        return Some(*def);
    }
    None
}

pub fn span_of_attrs(attrs: &Attributes) -> syntax_pos::Span {
    if attrs.doc_strings.is_empty() {
        return DUMMY_SP;
    }
    let start = attrs.doc_strings[0].span();
    let end = attrs.doc_strings.last().expect("No doc strings provided").span();
    start.to(end)
}

/// Reports a resolution failure diagnostic.
///
/// Ideally we can report the diagnostic with the actual span in the source where the link failure
/// occurred. However, there's a mismatch between the span in the source code and the span in the
/// markdown, so we have to do a bit of work to figure out the correspondence.
///
/// It's not too hard to find the span for sugared doc comments (`///` and `/**`), because the
/// source will match the markdown exactly, excluding the comment markers. However, it's much more
/// difficult to calculate the spans for unsugared docs, because we have to deal with escaping and
/// other source features. So, we attempt to find the exact source span of the resolution failure
/// in sugared docs, but use the span of the documentation attributes themselves for unsugared
/// docs. Because this span might be overly large, we display the markdown line containing the
/// failure as a note.
fn resolution_failure(
    cx: &DocContext,
    attrs: &Attributes,
    path_str: &str,
    dox: &str,
    link_range: Option<Range<usize>>,
) {
    let sp = span_of_attrs(attrs);
    let msg = format!("`[{}]` cannot be resolved, ignoring it...", path_str);

    let mut diag = if let Some(link_range) = link_range {
        let src = cx.sess().source_map().span_to_snippet(sp);
        let is_all_sugared_doc = attrs.doc_strings.iter().all(|frag| match frag {
            DocFragment::SugaredDoc(..) => true,
            _ => false,
        });

        if let (Ok(src), true) = (src, is_all_sugared_doc) {
            // The number of markdown lines up to and including the resolution failure.
            let num_lines = dox[..link_range.start].lines().count();

            // We use `split_terminator('\n')` instead of `lines()` when counting bytes to ensure
            // that DOS-style line endings do not cause the spans to be calculated incorrectly.
            let mut src_lines = src.split_terminator('\n');
            let mut md_lines = dox.split_terminator('\n').take(num_lines).peekable();

            // The number of bytes from the start of the source span to the resolution failure that
            // are *not* part of the markdown, like comment markers.
            let mut extra_src_bytes = 0;

            while let Some(md_line) = md_lines.next() {
                loop {
                    let source_line = src_lines
                        .next()
                        .expect("could not find markdown line in source");

                    match source_line.find(md_line) {
                        Some(offset) => {
                            extra_src_bytes += if md_lines.peek().is_some() {
                                source_line.len() - md_line.len()
                            } else {
                                offset
                            };
                            break;
                        }
                        None => {
                            // Since this is a source line that doesn't include a markdown line,
                            // we have to count the newline that we split from earlier.
                            extra_src_bytes += source_line.len() + 1;
                        }
                    }
                }
            }

            let sp = sp.from_inner_byte_pos(
                link_range.start + extra_src_bytes,
                link_range.end + extra_src_bytes,
            );

            let mut diag = cx.tcx.struct_span_lint_node(
                lint::builtin::INTRA_DOC_LINK_RESOLUTION_FAILURE,
                NodeId::from_u32(0),
                sp,
                &msg,
            );
            diag.span_label(sp, "cannot be resolved, ignoring");
            diag
        } else {
            let mut diag = cx.tcx.struct_span_lint_node(
                lint::builtin::INTRA_DOC_LINK_RESOLUTION_FAILURE,
                NodeId::from_u32(0),
                sp,
                &msg,
            );

            // blah blah blah\nblah\nblah [blah] blah blah\nblah blah
            //                       ^     ~~~~
            //                       |     link_range
            //                       last_new_line_offset
            let last_new_line_offset = dox[..link_range.start].rfind('\n').map_or(0, |n| n + 1);
            let line = dox[last_new_line_offset..].lines().next().unwrap_or("");

            // Print the line containing the `link_range` and manually mark it with '^'s.
            diag.note(&format!(
                "the link appears in this line:\n\n{line}\n\
                 {indicator: <before$}{indicator:^<found$}",
                line=line,
                indicator="",
                before=link_range.start - last_new_line_offset,
                found=link_range.len(),
            ));
            diag
        }
    } else {
        cx.tcx.struct_span_lint_node(lint::builtin::INTRA_DOC_LINK_RESOLUTION_FAILURE,
                                     NodeId::from_u32(0),
                                     sp,
                                     &msg)
    };
    diag.help("to escape `[` and `]` characters, just add '\\' before them like \
               `\\[` or `\\]`");
    diag.emit();
}

fn ambiguity_error(cx: &DocContext, attrs: &Attributes,
                   path_str: &str,
                   article1: &str, kind1: &str, disambig1: &str,
                   article2: &str, kind2: &str, disambig2: &str) {
    let sp = span_of_attrs(attrs);
    cx.sess()
      .struct_span_warn(sp,
                        &format!("`{}` is both {} {} and {} {}",
                                 path_str, article1, kind1,
                                 article2, kind2))
      .help(&format!("try `{}` if you want to select the {}, \
                      or `{}` if you want to \
                      select the {}",
                      disambig1, kind1, disambig2,
                      kind2))
      .emit();
}

/// Given a def, returns its name and disambiguator
/// for a value namespace.
///
/// Returns `None` for things which cannot be ambiguous since
/// they exist in both namespaces (structs and modules).
fn value_ns_kind(def: Def, path_str: &str) -> Option<(&'static str, String)> {
    match def {
        // Structs, variants, and mods exist in both namespaces; skip them.
        Def::StructCtor(..) | Def::Mod(..) | Def::Variant(..) |
        Def::VariantCtor(..) | Def::SelfCtor(..)
            => None,
        Def::Fn(..)
            => Some(("function", format!("{}()", path_str))),
        Def::Method(..)
            => Some(("method", format!("{}()", path_str))),
        Def::Const(..)
            => Some(("const", format!("const@{}", path_str))),
        Def::Static(..)
            => Some(("static", format!("static@{}", path_str))),
        _ => Some(("value", format!("value@{}", path_str))),
    }
}

/// Given a def, returns its name, the article to be used, and a disambiguator
/// for the type namespace.
fn type_ns_kind(def: Def, path_str: &str) -> (&'static str, &'static str, String) {
    let (kind, article) = match def {
        // We can still have non-tuple structs.
        Def::Struct(..) => ("struct", "a"),
        Def::Enum(..) => ("enum", "an"),
        Def::Trait(..) => ("trait", "a"),
        Def::Union(..) => ("union", "a"),
        _ => ("type", "a"),
    };
    (kind, article, format!("{}@{}", kind, path_str))
}

/// Given an enum variant's def, return the def of its enum and the associated fragment.
fn handle_variant(cx: &DocContext, def: Def) -> Result<(Def, Option<String>), ()> {
    use rustc::ty::DefIdTree;

    let parent = if let Some(parent) = cx.tcx.parent(def.def_id()) {
        parent
    } else {
        return Err(())
    };
    let parent_def = Def::Enum(parent);
    let variant = cx.tcx.expect_variant_def(def);
    Ok((parent_def, Some(format!("{}.v", variant.name))))
}

const PRIMITIVES: &[(&str, Def)] = &[
    ("u8",    Def::PrimTy(hir::PrimTy::Uint(syntax::ast::UintTy::U8))),
    ("u16",   Def::PrimTy(hir::PrimTy::Uint(syntax::ast::UintTy::U16))),
    ("u32",   Def::PrimTy(hir::PrimTy::Uint(syntax::ast::UintTy::U32))),
    ("u64",   Def::PrimTy(hir::PrimTy::Uint(syntax::ast::UintTy::U64))),
    ("u128",  Def::PrimTy(hir::PrimTy::Uint(syntax::ast::UintTy::U128))),
    ("usize", Def::PrimTy(hir::PrimTy::Uint(syntax::ast::UintTy::Usize))),
    ("i8",    Def::PrimTy(hir::PrimTy::Int(syntax::ast::IntTy::I8))),
    ("i16",   Def::PrimTy(hir::PrimTy::Int(syntax::ast::IntTy::I16))),
    ("i32",   Def::PrimTy(hir::PrimTy::Int(syntax::ast::IntTy::I32))),
    ("i64",   Def::PrimTy(hir::PrimTy::Int(syntax::ast::IntTy::I64))),
    ("i128",  Def::PrimTy(hir::PrimTy::Int(syntax::ast::IntTy::I128))),
    ("isize", Def::PrimTy(hir::PrimTy::Int(syntax::ast::IntTy::Isize))),
    ("f32",   Def::PrimTy(hir::PrimTy::Float(syntax::ast::FloatTy::F32))),
    ("f64",   Def::PrimTy(hir::PrimTy::Float(syntax::ast::FloatTy::F64))),
    ("str",   Def::PrimTy(hir::PrimTy::Str)),
    ("bool",  Def::PrimTy(hir::PrimTy::Bool)),
    ("char",  Def::PrimTy(hir::PrimTy::Char)),
];

fn is_primitive(path_str: &str, is_val: bool) -> Option<Def> {
    if is_val {
        None
    } else {
        PRIMITIVES.iter().find(|x| x.0 == path_str).map(|x| x.1)
    }
}
