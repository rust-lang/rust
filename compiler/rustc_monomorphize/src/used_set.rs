//! Cross-crate used-set consumption for `-Z dead-fn-used-set=<file>`.
//!
//! In its binary-only form (see [`super::dead_fn_elim`]) the pass is a no-op: a binary's
//! monomorphization collector is already exact, so nothing it emits is unreachable.
//!
//! The *cross-crate* form addresses the real slack. A library is compiled before its
//! callers and must conservatively emit its whole `pub` closure, most of which no final
//! binary calls. Once a binary is linked, the set of a dependency's functions it actually
//! reaches is *link-truth*: the defined function symbols present in the final executable
//! (`nm`). Feeding that set back on the library's (re)compilation via `-Zdead-fn-used-set`
//! lets the library keep only those symbols (plus what soundness requires) and drop the
//! rest before `partition()`.
//!
//! Identity is by **mangled symbol name**, matching link-truth exactly: a non-generic `pub`
//! function is codegen'd in the *dependency's* CGU and appears in the final binary as a
//! defined symbol, so `nm <binary>` names precisely the reachable set. (The binary's own
//! monomorphization collector does *not* list these — it only monomorphizes generics — which
//! is why the used-set must come from the link, not the collector.)
//!
//! File format (one mangled symbol per line, `#`-comments and blanks ignored):
//! ```text
//! # used-set for `deplib`, from `nm -j <binary>`
//! _RNvCseMk9t9oSu0C_6deplib8used_one
//! ```

use std::path::Path;

use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::def_id::DefId;

/// A parsed used-set: the mangled symbol names a final binary links from this crate.
pub(crate) struct UsedSet {
    symbols: FxHashSet<String>,
}

impl UsedSet {
    /// Parse a used-set file (one mangled symbol per line). Returns `None` (and warns) if the
    /// file cannot be read, so a missing/corrupt used-set degrades to the sound fallback of
    /// keeping the full `pub` closure rather than miscompiling.
    pub(crate) fn load(tcx: TyCtxt<'_>, path: &Path) -> Option<UsedSet> {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tcx.sess.dcx().warn(format!(
                    "-Z dead-fn-elimination: cannot read used-set file {}: {e}; \
                     keeping full public closure",
                    path.display()
                ));
                return None;
            }
        };
        let symbols: FxHashSet<String> = contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            // Take the first whitespace-delimited field, so raw `nm` output
            // (`<addr> T <sym>`) as well as bare-symbol lists both parse.
            .filter_map(|l| l.split_whitespace().last())
            .map(str::to_owned)
            .collect();
        Some(UsedSet { symbols })
    }

    /// Is this crate's function in the binary's used-set? Keyed on the item's mangled symbol
    /// name, which is what `nm` reports for the final binary.
    pub(crate) fn contains(&self, tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        // Only non-generic fns have a stable, callable mono symbol; generics are handled by
        // the collector and are never offered to `contains` (see the caller's guard).
        let instance = Instance::mono(tcx, def_id);
        let sym = MonoItem::Fn(instance).symbol_name(tcx).name;
        self.symbols.contains(sym)
    }

    pub(crate) fn len(&self) -> usize {
        self.symbols.len()
    }
}
