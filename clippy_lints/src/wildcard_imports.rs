use crate::utils::{in_macro, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc::ty::DefIdTree;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{walk_item, NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{symbol::Symbol, BytePos};

declare_clippy_lint! {
    /// **What it does:** Checks for wildcard imports `use _::*`.
    ///
    /// **Why is this bad?** wildcard imports can polute the namespace. This is especially bad if
    /// you try to import something through a wildcard, that already has been imported by name from
    /// a different source:
    ///
    /// ```rust,ignore
    /// use crate1::foo; // Imports a function named foo
    /// use crate2::*; // Has a function named foo
    ///
    /// foo(); // Calls crate1::foo
    /// ```
    ///
    /// This can lead to confusing error messages at best and to unexpected behavior at worst.
    ///
    /// **Known problems:** If macros are imported through the wildcard, this macro is not included
    /// by the suggestion and has to be added by hand.
    ///
    /// **Example:**
    ///
    /// Bad:
    /// ```rust,ignore
    /// use crate1::*;
    ///
    /// foo();
    /// ```
    ///
    /// Good:
    /// ```rust,ignore
    /// use crate1::foo;
    ///
    /// foo();
    /// ```
    pub WILDCARD_IMPORTS,
    pedantic,
    "lint `use _::*` statements"
}

declare_lint_pass!(WildcardImports => [WILDCARD_IMPORTS]);

impl LateLintPass<'_, '_> for WildcardImports {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, item: &Item<'_>) {
        if item.vis.node.is_pub() || item.vis.node.is_pub_restricted() {
            return;
        }
        if_chain! {
            if !in_macro(item.span);
            if let ItemKind::Use(use_path, UseKind::Glob) = &item.kind;
            if let Some(def_id) = use_path.res.opt_def_id();
            then {
                let hir = cx.tcx.hir();
                let parent_id = hir.get_parent_item(item.hir_id);
                let (items, in_module) = if parent_id == CRATE_HIR_ID {
                    let items = hir
                        .krate()
                        .module
                        .item_ids
                        .iter()
                        .map(|item_id| hir.get(item_id.id))
                        .filter_map(|node| {
                            if let Node::Item(item) = node {
                                Some(item)
                            } else {
                                None
                            }
                        })
                        .collect();
                    (items, true)
                } else if let Node::Item(item) = hir.get(parent_id) {
                    (vec![item], false)
                } else {
                    (vec![], false)
                };

                let mut import_used_visitor = ImportsUsedVisitor {
                    cx,
                    wildcard_def_id: def_id,
                    in_module,
                    used_imports: FxHashSet::default(),
                };
                for item in items {
                    import_used_visitor.visit_item(item);
                }

                if !import_used_visitor.used_imports.is_empty() {
                    let module_name = use_path
                        .segments
                        .iter()
                        .last()
                        .expect("path has at least one segment")
                        .ident
                        .name;

                    let mut applicability = Applicability::MachineApplicable;
                    let import_source = snippet_with_applicability(cx, use_path.span, "..", &mut applicability);
                    let (span, braced_glob) = if import_source.is_empty() {
                        // This is a `_::{_, *}` import
                        // Probably it's `_::{self, *}`, in that case we don't want to suggest to
                        // import `self`.
                        // If it is something else, we also don't want to include `self` in the
                        // suggestion, since either it was imported through another use statement:
                        // ```
                        // use foo::bar;
                        // use foo::bar::{baz, *};
                        // ```
                        // or it couldn't be used anywhere.
                        (
                            use_path.span.with_hi(use_path.span.hi() + BytePos(1)),
                            true,
                        )
                    } else {
                        (
                            use_path.span.with_hi(use_path.span.hi() + BytePos(3)),
                            false,
                        )
                    };

                    let imports_string = if import_used_visitor.used_imports.len() == 1 {
                        // We don't need to check for accidental suggesting the module name instead
                        // of `self` here, since if `used_imports.len() == 1`, and the only usage
                        // is `self`, then it's not through a `*` and if there is a `*`, it gets
                        // already linted by `unused_imports` of rustc.
                        import_used_visitor.used_imports.iter().next().unwrap().to_string()
                    } else {
                        let mut imports = import_used_visitor
                            .used_imports
                            .iter()
                            .filter_map(|import_name| {
                                if braced_glob && *import_name == module_name {
                                    None
                                } else if *import_name == module_name {
                                    Some("self".to_string())
                                } else {
                                    Some(import_name.to_string())
                                }
                            })
                            .collect::<Vec<_>>();
                        imports.sort();
                        if braced_glob {
                            imports.join(", ")
                        } else {
                            format!("{{{}}}", imports.join(", "))
                        }
                    };

                    let sugg = if import_source.is_empty() {
                        imports_string
                    } else {
                        format!("{}::{}", import_source, imports_string)
                    };

                    span_lint_and_sugg(
                        cx,
                        WILDCARD_IMPORTS,
                        span,
                        "usage of wildcard import",
                        "try",
                        sugg,
                        applicability,
                    );
                }
            }
        }
    }
}

struct ImportsUsedVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    wildcard_def_id: def_id::DefId,
    in_module: bool,
    used_imports: FxHashSet<Symbol>,
}

impl<'a, 'tcx> Visitor<'tcx> for ImportsUsedVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_item(&mut self, item: &'tcx Item<'_>) {
        match item.kind {
            ItemKind::Use(..) => {},
            ItemKind::Mod(..) if self.in_module => {},
            ItemKind::Mod(..) => self.in_module = true,
            _ => walk_item(self, item),
        }
    }

    fn visit_path(&mut self, path: &Path<'_>, _: HirId) {
        if let Some(def_id) = self.first_path_segment_def_id(path) {
            // Check if the function/enum/... was exported
            if let Some(exports) = self.cx.tcx.module_exports(self.wildcard_def_id) {
                for export in exports {
                    if let Some(export_def_id) = export.res.opt_def_id() {
                        if export_def_id == def_id {
                            self.used_imports.insert(
                                path.segments
                                    .iter()
                                    .next()
                                    .expect("path has at least one segment")
                                    .ident
                                    .name,
                            );
                            return;
                        }
                    }
                }
            }

            // Check if it is directly in the module
            if let Some(parent_def_id) = self.cx.tcx.parent(def_id) {
                if self.wildcard_def_id == parent_def_id {
                    self.used_imports.insert(
                        path.segments
                            .iter()
                            .next()
                            .expect("path has at least one segment")
                            .ident
                            .name,
                    );
                }
            }
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}

impl ImportsUsedVisitor<'_, '_> {
    fn skip_def_id(&self, def_id: DefId) -> DefId {
        let def_key = self.cx.tcx.def_key(def_id);
        match def_key.disambiguated_data.data {
            DefPathData::Ctor => {
                if let Some(def_id) = self.cx.tcx.parent(def_id) {
                    self.skip_def_id(def_id)
                } else {
                    def_id
                }
            },
            _ => def_id,
        }
    }

    fn first_path_segment_def_id(&self, path: &Path<'_>) -> Option<DefId> {
        path.res.opt_def_id().and_then(|mut def_id| {
            def_id = self.skip_def_id(def_id);
            for _ in path.segments.iter().skip(1) {
                def_id = self.skip_def_id(def_id);
                if let Some(parent_def_id) = self.cx.tcx.parent(def_id) {
                    def_id = parent_def_id;
                } else {
                    return None;
                }
            }

            Some(def_id)
        })
    }
}
