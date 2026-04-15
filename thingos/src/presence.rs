//! Canonical public types for the `thingos.presence` schema kind.
//!
//! # Schema (v1 — Deferred Embodiment Model)
//!
//! ```text
//! /// How an entity is present: the mode of its situated existence.
//! kind thingos.presence.mode = enum {
//!   Active,
//!   Latent,
//!   Remote,
//!   Projected,
//!   Deferred,
//!   Unavailable,
//! }
//!
//! /// The kind of embodiment describing how an entity is present.
//! kind thingos.presence.embodiment_kind = enum {
//!   Direct,
//!   Remote,
//!   Inherited,
//!   Symbolic,
//!   Deferred,
//! }
//!
//! /// Deferred embodiment model: situated existence of an entity within
//! /// the system's social/spatial context.
//! kind thingos.presence = struct {
//!   subject:     ref<thingos.person>,
//!   place:       option<ref<thingos.place>>,
//!   group:       option<ref<thingos.group>>,
//!   mode:        thingos.presence.mode,
//!   embodiment:  option<thingos.presence.embodiment_kind>,
//!   observed_at: option<u64>,
//! }
//! ```
//!
//! # What Presence is
//!
//! `Presence` is the canonical **deferred embodiment model** in ThingOS.  It
//! is a structured description of an entity's situated existence relative to
//! the system's social/spatial model.
//!
//! It answers questions like:
//! - present *where*? (via `place`)
//! - present in *what social context*? (via `group`)
//! - present *how*? (via `mode`)
//! - present *as what kind of embodiment*? (via `embodiment`)
//!
//! Presence is defined here as a **schema and conceptual boundary**, not a
//! live runtime system.  Runtime embodiment integration is deliberately
//! deferred until Place and Group foundations are stable.
//!
//! # Design principle
//!
//! > Presence should be defined as a stable descriptive schema before it
//! > becomes a runtime driver of behaviour.
//!
//! # What Presence is *not* (yet)
//!
//! * A scheduler primitive or execution residency concept.
//! * A full runtime capability or routing model.
//! * A live occupancy tracker or sensor-fusion result.
//! * An authority/permission boundary.
//! * A device, avatar, or body synchronisation system.
//! * A UI or shell presentation layer.
//! * A lifecycle coupling beyond descriptive schema references.
//!
//! # Relationship to Place and Group
//!
//! Presence may reference a [`Place`](crate::place::Place) without requiring
//! Place runtime semantics to be finished.  Presence may reference a
//! [`Group`](crate::group::Group) without forcing Group coordination behaviour
//! to be final.  Presence should not become a hidden Place/Group integration
//! layer.
//!
//! # Deferred integration points
//!
//! Likely future extension areas (not yet in scope):
//!
//! * Map Presence to Place/Group views once those are stable.
//! * Presence update/event flow over the messaging substrate.
//! * Connect Presence to UI/session representation.
//! * Explore interaction with authority and message routing.
//! * Refine embodiment kinds as device/avatar/body model matures.
//!
//! # Field rationale
//!
//! | Field         | Why it exists                                              |
//! |---------------|------------------------------------------------------------|
//! | `subject`     | Identifies *what* entity is present.                       |
//! | `place`       | Identifies *where* it is present (optional; may be unknown)|
//! | `group`       | Identifies the social/coordination context (optional)      |
//! | `mode`        | Describes the current quality/state of presence            |
//! | `embodiment`  | Lightweight descriptor of *how* embodied the presence is   |
//! | `observed_at` | When this presence record was last known valid             |

extern crate alloc;

/// The KindId generated for `thingos.presence` by `kindc`.
///
/// Identifies the canonical deferred-embodiment schema kind.  Consumers that
/// need to distinguish a `Presence`-shaped message payload can compare against
/// this constant.
pub const KIND_ID_THINGOS_PRESENCE: [u8; 16] = [
    0xde, 0xfd, 0xa1, 0x99, 0x59, 0x72, 0x8f, 0x20,
    0xaf, 0xbe, 0x09, 0xd0, 0x94, 0x72, 0xa4, 0xac,
];

/// The KindId generated for `thingos.presence.mode` by `kindc`.
pub const KIND_ID_THINGOS_PRESENCE_MODE: [u8; 16] = [
    0x9b, 0xaa, 0x44, 0x76, 0x79, 0xec, 0x7d, 0xbc,
    0xf3, 0x4f, 0x82, 0x74, 0xb6, 0xec, 0x96, 0xdf,
];

/// The KindId generated for `thingos.presence.embodiment_kind` by `kindc`.
pub const KIND_ID_THINGOS_PRESENCE_EMBODIMENT_KIND: [u8; 16] = [
    0x94, 0xa8, 0xe7, 0x82, 0x46, 0x4a, 0xb3, 0xd1,
    0xa6, 0x64, 0x78, 0xc2, 0x2f, 0x5d, 0x3b, 0x97,
];

/// How an entity is present: the mode of its situated existence.
///
/// Corresponds to the `thingos.presence.mode` schema kind (v1).
///
/// This enum is intentionally modest.  It describes the observable quality of
/// a presence record without committing to liveness guarantees or scheduling
/// semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresenceMode {
    /// The entity is actively present and reachable.
    Active,
    /// The entity is present but quiescent; not actively interacting.
    Latent,
    /// The entity is present through a remote or mediated connection.
    Remote,
    /// The entity is present through a derived or mirrored association.
    Projected,
    /// The entity's presence status has not yet been determined.
    ///
    /// Used when the system knows an entity *should* have a presence but
    /// has not yet established or confirmed it.
    Deferred,
    /// The entity cannot currently be reached or verified as present.
    Unavailable,
}

impl PresenceMode {
    /// Return a short human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            PresenceMode::Active => "Active",
            PresenceMode::Latent => "Latent",
            PresenceMode::Remote => "Remote",
            PresenceMode::Projected => "Projected",
            PresenceMode::Deferred => "Deferred",
            PresenceMode::Unavailable => "Unavailable",
        }
    }
}

/// The kind of embodiment describing how an entity is physically or
/// virtually present.
///
/// Corresponds to the `thingos.presence.embodiment_kind` schema kind (v1).
///
/// This is a lightweight descriptor, not a full embodiment model.  Full
/// device/sensor/avatar/body synchronisation is explicitly out of scope for
/// this phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbodimentKind {
    /// Direct, first-person, locally embodied presence.
    Direct,
    /// Presence mediated through a remote channel or proxy.
    Remote,
    /// Presence inherited from a parent or enclosing context.
    Inherited,
    /// A symbolic or representational placeholder presence.
    Symbolic,
    /// Embodiment is explicitly deferred pending future integration.
    Deferred,
}

impl EmbodimentKind {
    /// Return a short human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            EmbodimentKind::Direct => "Direct",
            EmbodimentKind::Remote => "Remote",
            EmbodimentKind::Inherited => "Inherited",
            EmbodimentKind::Symbolic => "Symbolic",
            EmbodimentKind::Deferred => "Deferred",
        }
    }
}

/// A 128-bit opaque reference to a subject entity (person or agent).
///
/// Wraps the raw `[u8; 16]` ThingId bytes produced by the `kindc` compiler
/// for `ref<thingos.person>` fields.  Kept opaque at this layer; higher-level
/// code resolves the referenced entity through the appropriate registry.
///
/// The deliberate opacity preserves the deferred nature of the Presence
/// schema: we name a subject without committing to how that subject is loaded
/// or resolved at runtime.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EntityRef(pub [u8; 16]);

/// A 128-bit opaque reference to a [`Place`](crate::place::Place).
///
/// Wraps the raw `[u8; 16]` ThingId bytes produced by the `kindc` compiler
/// for `ref<thingos.place>` fields.  Deferred: runtime Place lookup is not
/// yet required to construct a `Presence`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlaceRef(pub [u8; 16]);

/// A 128-bit opaque reference to a [`Group`](crate::group::Group).
///
/// Wraps the raw `[u8; 16]` ThingId bytes produced by the `kindc` compiler
/// for `ref<thingos.group>` fields.  Deferred: runtime Group lookup is not
/// yet required to construct a `Presence`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GroupRef(pub [u8; 16]);

/// Canonical deferred embodiment record for an entity in ThingOS.
///
/// Corresponds to the `thingos.presence` schema kind (v1).
///
/// A `Presence` describes *where* and *how* an entity is situated within the
/// system's social/spatial model.  It is a descriptive snapshot, not a live
/// runtime handle.  No runtime integration is performed in this phase.
///
/// # Deferred fields
///
/// * `place` and `group` are optional; a presence without a confirmed place
///   or group is valid (mode `Deferred` or `Unavailable` is appropriate).
/// * `embodiment` is optional; many presences may not carry a meaningful
///   embodiment descriptor yet.
/// * `observed_at` is optional; presence records may be constructed without
///   a known observation timestamp.
///
/// # Non-goals (deferred)
///
/// * Scheduling residency — not here.
/// * Automatic group joins or message-routing membership — not here.
/// * Authority/capability decisions — not here.
/// * Live sensor certainty or device/avatar synchronisation — not here.
/// * Physical pose, topology, or movement — not here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Presence {
    /// The entity whose presence this describes.
    ///
    /// A `ref<thingos.person>` (or future generic entity ref) encoded as an
    /// opaque 128-bit [`EntityRef`].  Resolution to a concrete entity type is
    /// deferred to the caller.
    pub subject: EntityRef,

    /// The place in which the entity is present, if known.
    ///
    /// References a [`Place`](crate::place::Place) by its ThingId.  `None`
    /// when the place is unknown, not yet established, or not applicable.
    /// Presence does not require Place runtime semantics to be complete.
    pub place: Option<PlaceRef>,

    /// The coordination group anchoring this presence, if any.
    ///
    /// References a [`Group`](crate::group::Group) by its ThingId.  `None`
    /// when the presence is not anchored to a coordination group.  Presence
    /// does not force Group coordination semantics to be final.
    pub group: Option<GroupRef>,

    /// How the presence currently exists.
    ///
    /// See [`PresenceMode`] for the available modes.
    pub mode: PresenceMode,

    /// The kind of embodiment, if applicable.
    ///
    /// `None` when embodiment is not yet determined or not meaningful for
    /// this presence record.  See [`EmbodimentKind`] for the available kinds.
    pub embodiment: Option<EmbodimentKind>,

    /// When this presence was last observed, in Unix milliseconds, if known.
    ///
    /// `None` when no observation timestamp is available.  This field is
    /// intentionally a raw `u64` at this layer; a richer timestamp type may
    /// be introduced once the system has a stable time abstraction.
    pub observed_at: Option<u64>,
}

impl Presence {
    /// Format as a human-readable text blob suitable for procfs or debugging.
    ///
    /// Output (all fields populated):
    /// ```text
    /// subject: 00000000000000000000000000000000
    /// place: 00000000000000000000000000000000
    /// group: none
    /// mode: Active
    /// embodiment: Direct
    /// observed_at: 1713139551000
    /// ```
    pub fn as_text(&self) -> alloc::string::String {
        let subject = hex_id(&self.subject.0);
        let place = self
            .place
            .map_or_else(|| alloc::string::String::from("none"), |p| hex_id(&p.0));
        let group = self
            .group
            .map_or_else(|| alloc::string::String::from("none"), |g| hex_id(&g.0));
        let embodiment = self
            .embodiment
            .map_or("none", |e| e.as_str());
        let observed = self
            .observed_at
            .map_or_else(|| alloc::string::String::from("none"), |t| alloc::format!("{}", t));

        alloc::format!(
            "subject: {}\nplace: {}\ngroup: {}\nmode: {}\nembodiment: {}\nobserved_at: {}\n",
            subject,
            place,
            group,
            self.mode.as_str(),
            embodiment,
            observed,
        )
    }
}

/// Format a 16-byte id as a lowercase hex string without separators.
fn hex_id(bytes: &[u8; 16]) -> alloc::string::String {
    let mut s = alloc::string::String::with_capacity(32);
    for b in bytes {
        let hi = (b >> 4) as usize;
        let lo = (b & 0xf) as usize;
        const HEX: &[u8] = b"0123456789abcdef";
        s.push(HEX[hi] as char);
        s.push(HEX[lo] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_entity() -> EntityRef {
        EntityRef([0u8; 16])
    }

    fn make_presence_active() -> Presence {
        Presence {
            subject: zero_entity(),
            place: None,
            group: None,
            mode: PresenceMode::Active,
            embodiment: None,
            observed_at: None,
        }
    }

    // ── PresenceMode ──────────────────────────────────────────────────────────

    #[test]
    fn test_presence_mode_as_str_active() {
        assert_eq!(PresenceMode::Active.as_str(), "Active");
    }

    #[test]
    fn test_presence_mode_as_str_latent() {
        assert_eq!(PresenceMode::Latent.as_str(), "Latent");
    }

    #[test]
    fn test_presence_mode_as_str_remote() {
        assert_eq!(PresenceMode::Remote.as_str(), "Remote");
    }

    #[test]
    fn test_presence_mode_as_str_projected() {
        assert_eq!(PresenceMode::Projected.as_str(), "Projected");
    }

    #[test]
    fn test_presence_mode_as_str_deferred() {
        assert_eq!(PresenceMode::Deferred.as_str(), "Deferred");
    }

    #[test]
    fn test_presence_mode_as_str_unavailable() {
        assert_eq!(PresenceMode::Unavailable.as_str(), "Unavailable");
    }

    #[test]
    fn test_presence_mode_equality() {
        assert_eq!(PresenceMode::Active, PresenceMode::Active);
        assert_ne!(PresenceMode::Active, PresenceMode::Latent);
    }

    // ── EmbodimentKind ────────────────────────────────────────────────────────

    #[test]
    fn test_embodiment_kind_as_str_direct() {
        assert_eq!(EmbodimentKind::Direct.as_str(), "Direct");
    }

    #[test]
    fn test_embodiment_kind_as_str_remote() {
        assert_eq!(EmbodimentKind::Remote.as_str(), "Remote");
    }

    #[test]
    fn test_embodiment_kind_as_str_inherited() {
        assert_eq!(EmbodimentKind::Inherited.as_str(), "Inherited");
    }

    #[test]
    fn test_embodiment_kind_as_str_symbolic() {
        assert_eq!(EmbodimentKind::Symbolic.as_str(), "Symbolic");
    }

    #[test]
    fn test_embodiment_kind_as_str_deferred() {
        assert_eq!(EmbodimentKind::Deferred.as_str(), "Deferred");
    }

    #[test]
    fn test_embodiment_kind_equality() {
        assert_eq!(EmbodimentKind::Direct, EmbodimentKind::Direct);
        assert_ne!(EmbodimentKind::Direct, EmbodimentKind::Symbolic);
    }

    // ── Presence construction ─────────────────────────────────────────────────

    #[test]
    fn test_presence_carries_subject() {
        let subject = EntityRef([0x01u8; 16]);
        let p = Presence {
            subject,
            place: None,
            group: None,
            mode: PresenceMode::Active,
            embodiment: None,
            observed_at: None,
        };
        assert_eq!(p.subject, subject);
    }

    #[test]
    fn test_presence_optional_place() {
        let place_ref = PlaceRef([0x02u8; 16]);
        let p = Presence {
            subject: zero_entity(),
            place: Some(place_ref),
            group: None,
            mode: PresenceMode::Active,
            embodiment: None,
            observed_at: None,
        };
        assert_eq!(p.place, Some(place_ref));
    }

    #[test]
    fn test_presence_place_absent() {
        let p = make_presence_active();
        assert_eq!(p.place, None);
    }

    #[test]
    fn test_presence_optional_group() {
        let group_ref = GroupRef([0x03u8; 16]);
        let p = Presence {
            subject: zero_entity(),
            place: None,
            group: Some(group_ref),
            mode: PresenceMode::Active,
            embodiment: None,
            observed_at: None,
        };
        assert_eq!(p.group, Some(group_ref));
    }

    #[test]
    fn test_presence_group_absent() {
        let p = make_presence_active();
        assert_eq!(p.group, None);
    }

    #[test]
    fn test_presence_carries_mode() {
        let p = Presence {
            subject: zero_entity(),
            place: None,
            group: None,
            mode: PresenceMode::Latent,
            embodiment: None,
            observed_at: None,
        };
        assert_eq!(p.mode, PresenceMode::Latent);
    }

    #[test]
    fn test_presence_optional_embodiment() {
        let p = Presence {
            subject: zero_entity(),
            place: None,
            group: None,
            mode: PresenceMode::Active,
            embodiment: Some(EmbodimentKind::Direct),
            observed_at: None,
        };
        assert_eq!(p.embodiment, Some(EmbodimentKind::Direct));
    }

    #[test]
    fn test_presence_embodiment_absent() {
        let p = make_presence_active();
        assert_eq!(p.embodiment, None);
    }

    #[test]
    fn test_presence_optional_observed_at() {
        let p = Presence {
            subject: zero_entity(),
            place: None,
            group: None,
            mode: PresenceMode::Active,
            embodiment: None,
            observed_at: Some(1_713_139_551_000),
        };
        assert_eq!(p.observed_at, Some(1_713_139_551_000));
    }

    #[test]
    fn test_presence_observed_at_absent() {
        let p = make_presence_active();
        assert_eq!(p.observed_at, None);
    }

    // ── equality ──────────────────────────────────────────────────────────────

    #[test]
    fn test_presence_equality_same() {
        let a = make_presence_active();
        let b = make_presence_active();
        assert_eq!(a, b);
    }

    #[test]
    fn test_presence_inequality_different_mode() {
        let a = make_presence_active();
        let b = Presence { mode: PresenceMode::Latent, ..make_presence_active() };
        assert_ne!(a, b);
    }

    #[test]
    fn test_presence_inequality_different_subject() {
        let a = Presence { subject: EntityRef([0x01u8; 16]), ..make_presence_active() };
        let b = Presence { subject: EntityRef([0x02u8; 16]), ..make_presence_active() };
        assert_ne!(a, b);
    }

    // ── as_text ───────────────────────────────────────────────────────────────

    #[test]
    fn test_as_text_contains_mode() {
        let p = make_presence_active();
        let text = p.as_text();
        assert!(text.contains("mode: Active"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_no_place_shows_none() {
        let p = make_presence_active();
        let text = p.as_text();
        assert!(text.contains("place: none"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_no_group_shows_none() {
        let p = make_presence_active();
        let text = p.as_text();
        assert!(text.contains("group: none"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_no_embodiment_shows_none() {
        let p = make_presence_active();
        let text = p.as_text();
        assert!(text.contains("embodiment: none"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_no_observed_at_shows_none() {
        let p = make_presence_active();
        let text = p.as_text();
        assert!(text.contains("observed_at: none"), "unexpected: {text}");
    }

    #[test]
    fn test_as_text_with_observed_at() {
        let p = Presence {
            observed_at: Some(1_000),
            ..make_presence_active()
        };
        assert!(p.as_text().contains("observed_at: 1000"));
    }

    #[test]
    fn test_as_text_ends_with_newline() {
        assert!(make_presence_active().as_text().ends_with('\n'));
    }

    #[test]
    fn test_as_text_contains_embodiment_when_set() {
        let p = Presence {
            embodiment: Some(EmbodimentKind::Symbolic),
            ..make_presence_active()
        };
        assert!(p.as_text().contains("embodiment: Symbolic"));
    }

    // ── clone ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_presence_clone_is_equal() {
        let a = make_presence_active();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── KIND_ID constants ─────────────────────────────────────────────────────

    #[test]
    fn test_kind_id_presence_constant() {
        let expected: [u8; 16] = [
            0xde, 0xfd, 0xa1, 0x99, 0x59, 0x72, 0x8f, 0x20,
            0xaf, 0xbe, 0x09, 0xd0, 0x94, 0x72, 0xa4, 0xac,
        ];
        assert_eq!(KIND_ID_THINGOS_PRESENCE, expected);
    }

    #[test]
    fn test_kind_id_presence_mode_constant() {
        let expected: [u8; 16] = [
            0x9b, 0xaa, 0x44, 0x76, 0x79, 0xec, 0x7d, 0xbc,
            0xf3, 0x4f, 0x82, 0x74, 0xb6, 0xec, 0x96, 0xdf,
        ];
        assert_eq!(KIND_ID_THINGOS_PRESENCE_MODE, expected);
    }

    #[test]
    fn test_kind_id_presence_embodiment_kind_constant() {
        let expected: [u8; 16] = [
            0x94, 0xa8, 0xe7, 0x82, 0x46, 0x4a, 0xb3, 0xd1,
            0xa6, 0x64, 0x78, 0xc2, 0x2f, 0x5d, 0x3b, 0x97,
        ];
        assert_eq!(KIND_ID_THINGOS_PRESENCE_EMBODIMENT_KIND, expected);
    }
}
