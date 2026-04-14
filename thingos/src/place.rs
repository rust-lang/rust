//! Canonical public types for the `thingos.place` schema kind.
//!
//! # Schema (v1 — Phase 8)
//!
//! ```text
//! /// A logical or physical location in which execution occurs
//! kind thingos.place = struct {
//!   cwd: string,
//!   namespace: string,
//!   root: string,
//! }
//! ```
//!
//! # What Place is
//!
//! `Place` is the canonical *world context* in ThingOS.  It is the explicit
//! answer to the question **"in what world does this execution occur?"**
//!
//! A Place captures the visibility boundaries and mount/namespace view that
//! define *where* a running unit lives, independent of *who* is running it
//! (which is `Authority`) or *how* it is coordinated (which is `Group`).
//!
//! In Phase 8 its shape is intentionally minimal.  The `cwd` records the
//! current working directory path.  The `namespace` labels the VFS mount
//! table view (currently always `"global"` because all processes share one
//! mount table).  The `root` records the effective filesystem root (always
//! `"/"` in Phase 8; per-process chroot is not yet implemented).
//!
//! # What belongs under Place
//!
//! * Current working directory
//! * VFS namespace / mount-table view
//! * Filesystem root (chroot-equivalent)
//! * Mount or view context that defines what paths resolve to
//! * Any other world-boundary state that answers "what can be seen from here"
//!
//! # What does NOT belong under Place (yet)
//!
//! * Terminal attachment, UI/console context — these belong to `Presence`
//!   (not yet introduced; see note below).
//! * Signal routing, coordination groups — these belong to `Group`.
//! * Permission checks, capability masks — these belong to `Authority`.
//! * Inherited Unix process-local environment blobs (`env`) — these are
//!   legacy compatibility state, quarantined in `Process::compat`.
//! * Session-like Unix assumptions (pgid, sid) — these are `Group` concerns
//!   quarantined as legacy compatibility until Phase 5+ extraction.
//!
//! # Note on Presence
//!
//! **Presence has not yet been introduced as a live execution/interaction
//! concept.**  This phase (Phase 8) intentionally isolates world-context
//! (Place) first.  Terminal attachment, UI attachment, and embodied
//! person-in-place relationships will be modelled under `Presence` in a
//! future phase.  Do not conflate the two.
//!
//! # Transitional mapping
//!
//! The current kernel `Process` structure is the *provisional* internal
//! backing for a `Place`.  The `kernel::place::bridge` module translates
//! `Process`-shaped world/context state into this public type.
//!
//! | Canonical field    | Current kernel source                        |
//! |--------------------|----------------------------------------------|
//! | `Place::cwd`       | `Process::cwd` (current working directory)   |
//! | `Place::namespace` | `Process::namespace` (always `"global"` now) |
//! | `Place::root`      | *(no per-process root yet — always `"/"`)    |
//!
//! # Future direction
//!
//! * Introduce per-process namespace isolation and populate `namespace` with
//!   a stable namespace identifier.
//! * Add per-process chroot / pivot-root support and populate `root`.
//! * Introduce `Presence` for terminal/UI/person-in-place relationships.
//! * Extend `Place` with mount-point listing or view-set membership.

extern crate alloc;

/// Canonical world-context for a running unit in ThingOS.
///
/// Corresponds to the `thingos.place` schema kind (v1).  The kernel's
/// internal `Process` structure is the current transitional backing; the
/// `bridge` module in `kernel::place` converts `Process`-shaped world/context
/// state into this type.
///
/// This struct is intentionally minimal for Phase 8.  Additional fields
/// (namespace identity, mount-set membership, chroot root) will be added as
/// the bridge matures and the internal `Process` place-context state is
/// progressively extracted.
///
/// # Note on Presence
///
/// Terminal attachment, UI attachment, and person-in-place relationships are
/// **not** represented here.  Those belong to `Presence`, which has not yet
/// been introduced as a live execution concept.  This type models only the
/// world/visibility context ("in what world"), not the embodied actor
/// relationship ("who is present in that world").
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Place {
    /// The current working directory path for this execution context.
    ///
    /// In Phase 8 this is derived directly from `Process::cwd`.  Future
    /// phases may replace this with a stable VFS-node reference once cwd
    /// tracking migrates out of raw path strings.
    pub cwd: alloc::string::String,

    /// The VFS namespace label for this execution context.
    ///
    /// In Phase 8 this is always `"global"` because all processes share a
    /// single global mount table (`NamespaceRef` is a unit struct today).
    /// Future phases will populate this with a stable namespace identifier
    /// once per-process namespace isolation is implemented.
    pub namespace: alloc::string::String,

    /// The effective filesystem root for this execution context.
    ///
    /// Always `"/"` in Phase 8; per-process chroot / pivot-root is not yet
    /// implemented.  Future phases will derive this from a per-process root
    /// binding once that mechanism is introduced.
    pub root: alloc::string::String,
}

impl Place {
    /// Format as a human-readable text blob suitable for procfs.
    ///
    /// Output:
    /// ```text
    /// cwd: /home/user
    /// namespace: global
    /// root: /
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        alloc::format!(
            "cwd: {}\nnamespace: {}\nroot: {}\n",
            self.cwd,
            self.namespace,
            self.root,
        )
    }
}
