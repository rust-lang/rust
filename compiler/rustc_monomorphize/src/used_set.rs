//! Cross-crate used-set: the producer ([`emit_used_sets`]) and consumer ([`UsedSet`]) for
//! `-Zdead-fn-emit-used-set` / `-Zdead-fn-used-set`.
//!
//! In its binary-only form (see [`super::dead_fn_elim`]) the pass is a no-op: a binary's
//! monomorphization collector is already exact, so nothing it emits is unreachable. The
//! *cross-crate* form addresses the real slack: a library is compiled before its callers and
//! must conservatively emit its whole `pub` closure, most of which no final binary calls.
//!
//! **Producer** ([`emit_used_sets`], run on the *binary* at `--emit=metadata`, before any
//! codegen): walk the binary's MIR and record every extern `fn` it references. This is the
//! set of a dependency's functions the binary actually reaches — computed without codegen, so
//! it can drive a *first-build* speedup. The binary's monomorphization collector cannot be
//! used here: it filters to local `DefId`s and never lists non-generic extern fns.
//!
//! **Consumer** ([`UsedSet`], run on each *library*): keep only the used-set functions (plus
//! what soundness requires) and drop the rest before `partition()`.
//!
//! **Identity** is the `local_hash` half of [`DefPathHash`] — the def-path fingerprint within
//! a crate. It is stable across compilations, unlike the mangled symbol name, whose crate
//! disambiguator differs between the probe compile and the dependency's own compile.
//!
//! File format (one 16-hex `local_hash` per line, `#`-comments and blanks ignored):
//! ```text
//! # used-set for `deplib` (MIR probe, DefPathHash local_hash)
//! 98912570aa6ef455
//! ```

use std::collections::BTreeMap;
use std::panic;
use std::path::Path;

use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::{DefId, LOCAL_CRATE};

/// A parsed used-set: the crate-local `DefPathHash` hashes a downstream binary reaches in
/// this crate. Keyed on the *local_hash* half of `DefPathHash`, which identifies an item by
/// its def-path within the crate and is stable across compilations (unlike the mangled
/// symbol name, whose crate-disambiguator changes between `cargo check` and `cargo build`).
pub(crate) struct UsedSet {
    local_hashes: FxHashSet<u64>,
}

impl UsedSet {
    /// Parse a used-set file (one 16-hex local_hash per line). Returns `None` (and warns) if
    /// the file cannot be read, so a missing/corrupt used-set degrades to the sound fallback
    /// of keeping the full `pub` closure rather than miscompiling.
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
        let local_hashes: FxHashSet<u64> = contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| u64::from_str_radix(l, 16).ok())
            .collect();
        Some(UsedSet { local_hashes })
    }

    /// Is this crate's function in the binary's used-set? Keyed on the def-path-stable
    /// `local_hash`, so it matches regardless of the crate disambiguator differences between
    /// the probe compile and this one.
    pub(crate) fn contains(&self, tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        self.local_hashes.contains(&tcx.def_path_hash(def_id).local_hash().as_u64())
    }

    pub(crate) fn len(&self) -> usize {
        self.local_hashes.len()
    }
}

/// The used-set **probe** (component 1). Walks this crate's MIR — available after analysis,
/// *before* codegen — and records every *extern* (non-local) function it references from a
/// call or as a function value. Each such reference is the exact non-generic dependency
/// function this crate reaches; grouped by crate and written as one `<crate>.usedset` file
/// per dependency into `dir`.
///
/// This is the first-build mechanism: it runs on `--emit=metadata` (no codegen), so a later
/// dependency compile can be pruned against the used-set and codegen'd *once*, rather than
/// fully codegen'd and then discarded. It deliberately does **not** use the monomorphization
/// collector, which filters to local `DefId`s and so never lists non-generic extern fns.
pub(crate) fn emit_used_sets(tcx: TyCtxt<'_>, dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tcx.sess
            .dcx()
            .warn(format!("-Z dead-fn-emit-used-set: cannot create {}: {e}", dir.display()));
        return;
    }

    // Collect extern fn DefIds referenced across all local MIR bodies.
    let mut externs: FxIndexSet<DefId> = FxIndexSet::default();
    for &local_def_id in tcx.mir_keys(()) {
        let def_id = local_def_id.to_def_id();
        if !tcx.is_mir_available(def_id) {
            continue;
        }
        let Ok(body) = panic::catch_unwind(panic::AssertUnwindSafe(|| tcx.optimized_mir(def_id)))
        else {
            continue;
        };
        for bb in body.basic_blocks.iter() {
            if let TerminatorKind::Call { func, .. } = &bb.terminator().kind
                && let rustc_middle::mir::Operand::Constant(c) = func
                && let ty::FnDef(callee, _) = c.const_.ty().kind()
                && !callee.is_local()
            {
                externs.insert(*callee);
            }
            for stmt in &bb.statements {
                use rustc_middle::mir::{Rvalue, StatementKind};
                if let StatementKind::Assign(box (_, Rvalue::Use(op, _) | Rvalue::Cast(_, op, _))) =
                    &stmt.kind
                    && let rustc_middle::mir::Operand::Constant(c) = op
                    && let ty::FnDef(callee, _) = c.const_.ty().kind()
                    && !callee.is_local()
                {
                    externs.insert(*callee);
                }
            }
        }
    }

    // Group by defining crate; key each entry by the extern fn's def-path-stable `local_hash`
    // (the low half of its `DefPathHash`). This is stable across compilations — unlike the
    // mangled symbol, whose crate disambiguator differs between the probe compile and the
    // dependency's own compile — so the consumer matches it reliably.
    let mut per_crate: BTreeMap<String, std::collections::BTreeSet<u64>> = BTreeMap::new();
    for &did in &externs {
        if did.krate == LOCAL_CRATE || tcx.def_kind(did) != rustc_hir::def::DefKind::Fn {
            continue;
        }
        // Only non-generic fns are eliminable candidates; generics are handled by the collector.
        if tcx.generics_of(did).count() != 0 {
            continue;
        }
        let name = tcx.crate_name(did.krate).to_string();
        let local_hash = tcx.def_path_hash(did).local_hash().as_u64();
        per_crate.entry(name).or_default().insert(local_hash);
    }

    // The used-set of a dependency is the *union* over every crate that calls it: `Main`'s
    // probe names `a::f`, but `b::used` is called from inside `a`'s body, so only `a`'s probe
    // names it. Each crate therefore *appends* its references; duplicate lines are harmless
    // (the consumer dedups into a set). Append is used rather than read-modify-write so the
    // parallel crate compilations writing the same file do not race.
    use std::io::Write;
    let mut wrote = 0;
    for (crate_name, hashes) in &per_crate {
        let path = dir.join(format!("{crate_name}.usedset"));
        let mut body = String::new();
        for h in hashes {
            body.push_str(&format!("{h:016x}\n"));
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            && f.write_all(body.as_bytes()).is_ok()
        {
            wrote += 1;
        }
    }
    tcx.sess.dcx().note(format!(
        "-Z dead-fn-emit-used-set: probed {} extern fns → {} used-set files in {}",
        externs.len(),
        wrote,
        dir.display()
    ));
}
