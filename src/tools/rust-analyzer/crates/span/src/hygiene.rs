//! Machinery for hygienic macros.
//!
//! Inspired by Matthew Flatt et al., “Macros That Work Together: Compile-Time Bindings, Partial
//! Expansion, and Definition Contexts,” *Journal of Functional Programming* 22, no. 2
//! (March 1, 2012): 181–216, <https://doi.org/10.1017/S0956796812000093>.
//!
//! Also see <https://rustc-dev-guide.rust-lang.org/macro-expansion.html#hygiene-and-hierarchies>
//!
//! # The Expansion Order Hierarchy
//!
//! `ExpnData` in rustc, rust-analyzer's version is [`MacroCallLoc`]. Traversing the hierarchy
//! upwards can be achieved by walking up [`MacroCallLoc::kind`]'s contained file id, as
//! [`MacroFile`]s are interned [`MacroCallLoc`]s.
//!
//! # The Macro Definition Hierarchy
//!
//! `SyntaxContextData` in rustc and rust-analyzer. Basically the same in both.
//!
//! # The Call-site Hierarchy
//!
//! `ExpnData::call_site` in rustc, [`MacroCallLoc::call_site`] in rust-analyzer.
use std::fmt;

use crate::Edition;

/// A syntax context describes a hierarchy tracking order of macro definitions.
#[cfg(feature = "salsa")]
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct SyntaxContext(
    /// # Invariant
    ///
    /// This is either a valid `salsa::Id` or a root `SyntaxContext`.
    u32,
    std::marker::PhantomData<&'static salsa::plumbing::interned::Value<SyntaxContext>>,
);

#[cfg(feature = "salsa")]
const _: () = {
    use crate::MacroCallId;
    use salsa::plumbing as zalsa_;
    use salsa::plumbing::interned as zalsa_struct_;

    #[derive(Clone, Eq, Debug)]
    pub struct SyntaxContextData {
        outer_expn: Option<MacroCallId>,
        outer_transparency: Transparency,
        edition: Edition,
        parent: SyntaxContext,
        opaque: SyntaxContext,
        opaque_and_semitransparent: SyntaxContext,
    }

    impl PartialEq for SyntaxContextData {
        fn eq(&self, other: &Self) -> bool {
            self.outer_expn == other.outer_expn
                && self.outer_transparency == other.outer_transparency
                && self.edition == other.edition
                && self.parent == other.parent
        }
    }

    impl std::hash::Hash for SyntaxContextData {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.outer_expn.hash(state);
            self.outer_transparency.hash(state);
            self.edition.hash(state);
            self.parent.hash(state);
        }
    }
    /// Key to use during hash lookups. Each field is some type that implements `Lookup<T>`
    /// for the owned type. This permits interning with an `&str` when a `String` is required and so forth.
    #[derive(Hash)]
    struct StructKey<'db, T0, T1, T2, T3>(T0, T1, T2, T3, std::marker::PhantomData<&'db ()>);

    impl<'db, T0, T1, T2, T3> zalsa_::interned::HashEqLike<StructKey<'db, T0, T1, T2, T3>>
        for SyntaxContextData
    where
        Option<MacroCallId>: zalsa_::interned::HashEqLike<T0>,
        Transparency: zalsa_::interned::HashEqLike<T1>,
        Edition: zalsa_::interned::HashEqLike<T2>,
        SyntaxContext: zalsa_::interned::HashEqLike<T3>,
    {
        fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
            zalsa_::interned::HashEqLike::<T0>::hash(&self.outer_expn, &mut *h);
            zalsa_::interned::HashEqLike::<T1>::hash(&self.outer_transparency, &mut *h);
            zalsa_::interned::HashEqLike::<T2>::hash(&self.edition, &mut *h);
            zalsa_::interned::HashEqLike::<T3>::hash(&self.parent, &mut *h);
        }
        fn eq(&self, data: &StructKey<'db, T0, T1, T2, T3>) -> bool {
            zalsa_::interned::HashEqLike::<T0>::eq(&self.outer_expn, &data.0)
                && zalsa_::interned::HashEqLike::<T1>::eq(&self.outer_transparency, &data.1)
                && zalsa_::interned::HashEqLike::<T2>::eq(&self.edition, &data.2)
                && zalsa_::interned::HashEqLike::<T3>::eq(&self.parent, &data.3)
        }
    }
    impl zalsa_struct_::Configuration for SyntaxContext {
        const LOCATION: salsa::plumbing::Location =
            salsa::plumbing::Location { file: file!(), line: line!() };
        const DEBUG_NAME: &'static str = "SyntaxContextData";
        const REVISIONS: std::num::NonZeroUsize = std::num::NonZeroUsize::MAX;
        type Fields<'a> = SyntaxContextData;
        type Struct<'a> = SyntaxContext;
    }
    impl SyntaxContext {
        pub fn ingredient<Db>(db: &Db) -> &zalsa_struct_::IngredientImpl<Self>
        where
            Db: ?Sized + zalsa_::Database,
        {
            static CACHE: zalsa_::IngredientCache<zalsa_struct_::IngredientImpl<SyntaxContext>> =
                zalsa_::IngredientCache::new();
            CACHE.get_or_create(db.zalsa(), || {
                db.zalsa()
                    .lookup_jar_by_type::<zalsa_struct_::JarImpl<SyntaxContext>>()
                    .get_or_create()
            })
        }
    }
    impl zalsa_::AsId for SyntaxContext {
        fn as_id(&self) -> salsa::Id {
            self.as_salsa_id().expect("`SyntaxContext::as_id()` called on a root `SyntaxContext`")
        }
    }
    impl zalsa_::FromId for SyntaxContext {
        fn from_id(id: salsa::Id) -> Self {
            Self::from_salsa_id(id)
        }
    }
    unsafe impl Send for SyntaxContext {}

    unsafe impl Sync for SyntaxContext {}

    impl zalsa_::SalsaStructInDb for SyntaxContext {
        type MemoIngredientMap = salsa::plumbing::MemoIngredientSingletonIndex;

        fn lookup_or_create_ingredient_index(
            zalsa: &salsa::plumbing::Zalsa,
        ) -> salsa::plumbing::IngredientIndices {
            zalsa
                .lookup_jar_by_type::<zalsa_struct_::JarImpl<SyntaxContext>>()
                .get_or_create()
                .into()
        }

        #[inline]
        fn cast(id: salsa::Id, type_id: std::any::TypeId) -> Option<Self> {
            if type_id == std::any::TypeId::of::<SyntaxContext>() {
                Some(<Self as salsa::plumbing::FromId>::from_id(id))
            } else {
                None
            }
        }
    }

    unsafe impl salsa::plumbing::Update for SyntaxContext {
        unsafe fn maybe_update(old_pointer: *mut Self, new_value: Self) -> bool {
            if unsafe { *old_pointer } != new_value {
                unsafe { *old_pointer = new_value };
                true
            } else {
                false
            }
        }
    }
    impl<'db> SyntaxContext {
        pub fn new<
            Db,
            T0: zalsa_::interned::Lookup<Option<MacroCallId>> + std::hash::Hash,
            T1: zalsa_::interned::Lookup<Transparency> + std::hash::Hash,
            T2: zalsa_::interned::Lookup<Edition> + std::hash::Hash,
            T3: zalsa_::interned::Lookup<SyntaxContext> + std::hash::Hash,
        >(
            db: &'db Db,
            outer_expn: T0,
            outer_transparency: T1,
            edition: T2,
            parent: T3,
            opaque: impl FnOnce(SyntaxContext) -> SyntaxContext,
            opaque_and_semitransparent: impl FnOnce(SyntaxContext) -> SyntaxContext,
        ) -> Self
        where
            Db: ?Sized + salsa::Database,
            Option<MacroCallId>: zalsa_::interned::HashEqLike<T0>,
            Transparency: zalsa_::interned::HashEqLike<T1>,
            Edition: zalsa_::interned::HashEqLike<T2>,
            SyntaxContext: zalsa_::interned::HashEqLike<T3>,
        {
            SyntaxContext::ingredient(db).intern(
                db.as_dyn_database(),
                StructKey::<'db>(
                    outer_expn,
                    outer_transparency,
                    edition,
                    parent,
                    std::marker::PhantomData,
                ),
                |id, data| SyntaxContextData {
                    outer_expn: zalsa_::interned::Lookup::into_owned(data.0),
                    outer_transparency: zalsa_::interned::Lookup::into_owned(data.1),
                    edition: zalsa_::interned::Lookup::into_owned(data.2),
                    parent: zalsa_::interned::Lookup::into_owned(data.3),
                    opaque: opaque(zalsa_::FromId::from_id(id)),
                    opaque_and_semitransparent: opaque_and_semitransparent(
                        zalsa_::FromId::from_id(id),
                    ),
                },
            )
        }

        /// Invariant: Only [`SyntaxContext::ROOT`] has a [`None`] outer expansion.
        // FIXME: The None case needs to encode the context crate id. We can encode that as the MSB of
        // MacroCallId is reserved anyways so we can do bit tagging here just fine.
        // The bigger issue is that this will cause interning to now create completely separate chains
        // per crate. Though that is likely not a problem as `MacroCallId`s are already crate calling dependent.
        pub fn outer_expn<Db>(self, db: &'db Db) -> Option<MacroCallId>
        where
            Db: ?Sized + zalsa_::Database,
        {
            let id = self.as_salsa_id()?;
            let fields = SyntaxContext::ingredient(db).data(db.as_dyn_database(), id);
            fields.outer_expn
        }

        pub fn outer_transparency<Db>(self, db: &'db Db) -> Transparency
        where
            Db: ?Sized + zalsa_::Database,
        {
            let Some(id) = self.as_salsa_id() else { return Transparency::Opaque };
            let fields = SyntaxContext::ingredient(db).data(db.as_dyn_database(), id);
            fields.outer_transparency
        }

        pub fn edition<Db>(self, db: &'db Db) -> Edition
        where
            Db: ?Sized + zalsa_::Database,
        {
            match self.as_salsa_id() {
                Some(id) => {
                    let fields = SyntaxContext::ingredient(db).data(db.as_dyn_database(), id);
                    fields.edition
                }
                None => Edition::from_u32(SyntaxContext::MAX_ID - self.into_u32()),
            }
        }

        pub fn parent<Db>(self, db: &'db Db) -> SyntaxContext
        where
            Db: ?Sized + zalsa_::Database,
        {
            match self.as_salsa_id() {
                Some(id) => {
                    let fields = SyntaxContext::ingredient(db).data(db.as_dyn_database(), id);
                    fields.parent
                }
                None => self,
            }
        }

        /// This context, but with all transparent and semi-transparent expansions filtered away.
        pub fn opaque<Db>(self, db: &'db Db) -> SyntaxContext
        where
            Db: ?Sized + zalsa_::Database,
        {
            match self.as_salsa_id() {
                Some(id) => {
                    let fields = SyntaxContext::ingredient(db).data(db.as_dyn_database(), id);
                    fields.opaque
                }
                None => self,
            }
        }

        /// This context, but with all transparent expansions filtered away.
        pub fn opaque_and_semitransparent<Db>(self, db: &'db Db) -> SyntaxContext
        where
            Db: ?Sized + zalsa_::Database,
        {
            match self.as_salsa_id() {
                Some(id) => {
                    let fields = SyntaxContext::ingredient(db).data(db.as_dyn_database(), id);
                    fields.opaque_and_semitransparent
                }
                None => self,
            }
        }
    }
};

impl SyntaxContext {
    #[inline]
    pub fn is_root(self) -> bool {
        (SyntaxContext::MAX_ID - Edition::LATEST as u32) <= self.into_u32()
            && self.into_u32() <= (SyntaxContext::MAX_ID - Edition::Edition2015 as u32)
    }

    #[inline]
    pub fn remove_root_edition(&mut self) {
        if self.is_root() {
            *self = Self::root(Edition::Edition2015);
        }
    }

    /// The root context, which is the parent of all other contexts. All [`FileId`]s have this context.
    #[inline]
    pub const fn root(edition: Edition) -> Self {
        let edition = edition as u32;
        // SAFETY: Roots are valid `SyntaxContext`s
        unsafe { SyntaxContext::from_u32(SyntaxContext::MAX_ID - edition) }
    }
}

#[cfg(feature = "salsa")]
impl<'db> SyntaxContext {
    const MAX_ID: u32 = salsa::Id::MAX_U32 - 1;

    #[inline]
    pub const fn into_u32(self) -> u32 {
        self.0
    }

    /// # Safety
    ///
    /// The ID must be a valid `SyntaxContext`.
    #[inline]
    pub const unsafe fn from_u32(u32: u32) -> Self {
        // INVARIANT: Our precondition.
        Self(u32, std::marker::PhantomData)
    }

    #[inline]
    fn as_salsa_id(self) -> Option<salsa::Id> {
        if self.is_root() {
            None
        } else {
            // SAFETY: By our invariant, this is either a root (which we verified it's not) or a valid `salsa::Id`.
            unsafe { Some(salsa::Id::from_index(self.0)) }
        }
    }

    #[inline]
    fn from_salsa_id(id: salsa::Id) -> Self {
        // SAFETY: This comes from a Salsa ID.
        unsafe { Self::from_u32(id.index()) }
    }

    #[inline]
    pub fn outer_mark(
        self,
        db: &'db dyn salsa::Database,
    ) -> (Option<crate::MacroCallId>, Transparency) {
        (self.outer_expn(db), self.outer_transparency(db))
    }

    #[inline]
    pub fn normalize_to_macros_2_0(self, db: &'db dyn salsa::Database) -> SyntaxContext {
        self.opaque(db)
    }

    #[inline]
    pub fn normalize_to_macro_rules(self, db: &'db dyn salsa::Database) -> SyntaxContext {
        self.opaque_and_semitransparent(db)
    }

    pub fn is_opaque(self, db: &'db dyn salsa::Database) -> bool {
        !self.is_root() && self.outer_transparency(db).is_opaque()
    }

    pub fn remove_mark(
        &mut self,
        db: &'db dyn salsa::Database,
    ) -> (Option<crate::MacroCallId>, Transparency) {
        let data = *self;
        *self = data.parent(db);
        (data.outer_expn(db), data.outer_transparency(db))
    }

    pub fn marks(
        self,
        db: &'db dyn salsa::Database,
    ) -> impl Iterator<Item = (crate::MacroCallId, Transparency)> {
        let mut marks = self.marks_rev(db).collect::<Vec<_>>();
        marks.reverse();
        marks.into_iter()
    }

    pub fn marks_rev(
        self,
        db: &'db dyn salsa::Database,
    ) -> impl Iterator<Item = (crate::MacroCallId, Transparency)> {
        std::iter::successors(Some(self), move |&mark| Some(mark.parent(db)))
            .take_while(|&it| !it.is_root())
            .map(|ctx| {
                let mark = ctx.outer_mark(db);
                // We stop before taking the root expansion, as such we cannot encounter a `None` outer
                // expansion, as only the ROOT has it.
                (mark.0.unwrap(), mark.1)
            })
    }
}
#[cfg(not(feature = "salsa"))]
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct SyntaxContext(u32);

#[allow(dead_code)]
const SALSA_MAX_ID_MIRROR: u32 = u32::MAX - 0xFF;
#[cfg(feature = "salsa")]
const _: () = assert!(salsa::Id::MAX_U32 == SALSA_MAX_ID_MIRROR);

#[cfg(not(feature = "salsa"))]
impl SyntaxContext {
    const MAX_ID: u32 = SALSA_MAX_ID_MIRROR - 1;

    pub const fn into_u32(self) -> u32 {
        self.0
    }

    /// # Safety
    ///
    /// None. This is always safe to call without the `salsa` feature.
    pub const unsafe fn from_u32(u32: u32) -> Self {
        Self(u32)
    }
}

/// A property of a macro expansion that determines how identifiers
/// produced by that expansion are resolved.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug)]
pub enum Transparency {
    /// Identifier produced by a transparent expansion is always resolved at call-site.
    /// Call-site spans in procedural macros, hygiene opt-out in `macro` should use this.
    Transparent,
    /// Identifier produced by a semi-transparent expansion may be resolved
    /// either at call-site or at definition-site.
    /// If it's a local variable, label or `$crate` then it's resolved at def-site.
    /// Otherwise it's resolved at call-site.
    /// `macro_rules` macros behave like this, built-in macros currently behave like this too,
    /// but that's an implementation detail.
    SemiTransparent,
    /// Identifier produced by an opaque expansion is always resolved at definition-site.
    /// Def-site spans in procedural macros, identifiers from `macro` by default use this.
    Opaque,
}

impl Transparency {
    /// Returns `true` if the transparency is [`Opaque`].
    ///
    /// [`Opaque`]: Transparency::Opaque
    pub fn is_opaque(&self) -> bool {
        matches!(self, Self::Opaque)
    }
}

impl fmt::Display for SyntaxContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_root() {
            write!(f, "ROOT{}", Edition::from_u32(SyntaxContext::MAX_ID - self.into_u32()).number())
        } else {
            write!(f, "{}", self.into_u32())
        }
    }
}

impl std::fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            fmt::Display::fmt(self, f)
        } else {
            f.debug_tuple("SyntaxContext").field(&self.0).finish()
        }
    }
}
