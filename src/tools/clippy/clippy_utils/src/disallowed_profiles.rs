use crate::sym;
use rustc_ast::ast::{LitKind, MetaItemInner};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::smallvec::SmallVec;
use rustc_hir::{Attribute, HirId};
use rustc_lint::LateContext;
use rustc_span::{Span, Symbol};

/// One profile name referenced by a `#[clippy::disallowed_profile(...)]` or
/// `#[clippy::disallowed_profiles(...)]` attribute on an item.
///
/// A single attribute produces one `ProfileEntry` per string argument. The entry records which
/// attribute variant introduced it, the profile name, and the span of that string literal so
/// diagnostics (e.g. "unknown profile") can point at the exact argument.
#[derive(Copy, Clone)]
pub struct ProfileEntry {
    pub attr_name: Symbol,
    pub name: Symbol,
    pub span: Span,
}

/// The set of profiles active at some `HirId`, obtained by walking up the HIR from that id and
/// collecting the first ancestor that carries a `#[clippy::disallowed_profile(s)]` attribute.
///
/// An empty selection is represented by `None` at the call site; a `ProfileSelection` is always
/// non-empty.
#[derive(Clone)]
pub struct ProfileSelection {
    entries: SmallVec<[ProfileEntry; 2]>,
}

impl ProfileSelection {
    pub fn new(entries: SmallVec<[ProfileEntry; 2]>) -> Self {
        Self { entries }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ProfileEntry> {
        self.entries.iter()
    }
}

#[derive(Default)]
pub struct ProfileResolver {
    cache: FxHashMap<HirId, Option<ProfileSelection>>,
}

impl ProfileResolver {
    pub fn active_profiles(&mut self, cx: &LateContext<'_>, hir_id: HirId) -> Option<&ProfileSelection> {
        // NOTE: The `contains_key`+`get` dance is intentional: using only `get` here triggers borrowck
        // errors because we need to mutate `self.cache` on cache misses.
        if self.cache.contains_key(&hir_id) {
            return self.cache.get(&hir_id).and_then(|selection| selection.as_ref());
        }

        let (resolved, visited) = self.resolve(cx, hir_id);

        for id in visited {
            self.cache.entry(id).or_insert_with(|| resolved.clone());
        }
        self.cache.insert(hir_id, resolved);

        self.cache.get(&hir_id).and_then(|selection| selection.as_ref())
    }

    fn resolve(&self, cx: &LateContext<'_>, start: HirId) -> (Option<ProfileSelection>, SmallVec<[HirId; 8]>) {
        let mut visited = SmallVec::<[HirId; 8]>::new();
        let mut current = Some(start);

        while let Some(id) = current {
            if let Some(cached) = self.cache.get(&id) {
                return (cached.clone(), visited);
            }

            visited.push(id);

            if let Some(selection) = profiles_from_attrs(cx, cx.tcx.hir_attrs(id)) {
                return (Some(selection), visited);
            }

            if id == rustc_hir::CRATE_HIR_ID {
                current = None;
            } else {
                current = Some(cx.tcx.parent_hir_id(id));
            }
        }

        (None, visited)
    }
}

fn profiles_from_attrs(cx: &LateContext<'_>, attrs: &[Attribute]) -> Option<ProfileSelection> {
    let mut entries = SmallVec::<[ProfileEntry; 2]>::new();

    for attr in attrs {
        let path = attr.path();
        if path.len() != 2 || path[0] != sym::clippy {
            continue;
        }

        let name = path[1];
        if name != sym::disallowed_profile && name != sym::disallowed_profiles {
            continue;
        }

        let attr_label = if name == sym::disallowed_profiles {
            "`clippy::disallowed_profiles`"
        } else {
            "`clippy::disallowed_profile`"
        };

        let Some(items) = attr.meta_item_list() else {
            cx.tcx
                .sess
                .dcx()
                .struct_span_err(attr.span(), format!("{attr_label} expects string arguments"))
                .emit();
            continue;
        };

        if items.is_empty() {
            cx.tcx
                .sess
                .dcx()
                .struct_span_err(attr.span(), format!("{attr_label} expects at least one profile name"))
                .emit();
            continue;
        }

        if name == sym::disallowed_profile && items.len() != 1 {
            cx.tcx
                .sess
                .dcx()
                .struct_span_err(attr.span(), "use `clippy::disallowed_profiles` for multiple profiles")
                .emit();
        }

        for item in items {
            match literal_symbol(&item) {
                Some((symbol, span)) => entries.push(ProfileEntry {
                    attr_name: name,
                    name: symbol,
                    span,
                }),
                None => emit_string_error(cx, &item),
            }
        }
    }

    if entries.is_empty() {
        None
    } else {
        Some(ProfileSelection::new(entries))
    }
}

fn literal_symbol(item: &MetaItemInner) -> Option<(Symbol, Span)> {
    match item {
        MetaItemInner::Lit(lit) => {
            let LitKind::Str(symbol, _) = lit.kind else { return None };
            Some((symbol, lit.span))
        },
        MetaItemInner::MetaItem(_) => None,
    }
}

fn emit_string_error(cx: &LateContext<'_>, item: &MetaItemInner) {
    let span = match item {
        MetaItemInner::Lit(lit) => lit.span,
        MetaItemInner::MetaItem(meta) => meta.span,
    };
    cx.tcx
        .sess
        .dcx()
        .struct_span_err(span, "expected string literal profile name")
        .emit();
}
