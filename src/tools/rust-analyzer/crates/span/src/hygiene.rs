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

use crate::{Edition, MacroCallId};

// Recursive expansion of interned macro
// ======================================

/// A syntax context describes a hierarchy tracking order of macro definitions.
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct SyntaxContext(
    salsa::Id,
    std::marker::PhantomData<&'static salsa::plumbing::interned::Value<SyntaxContext>>,
);

/// The underlying data interned by Salsa.
#[derive(Clone, Eq, Debug)]
pub struct SyntaxContextUnderlyingData {
    pub outer_expn: Option<MacroCallId>,
    pub outer_transparency: Transparency,
    pub edition: Edition,
    pub parent: SyntaxContext,
    pub opaque: SyntaxContext,
    pub opaque_and_semitransparent: SyntaxContext,
}

const _: () = {
    use salsa::plumbing as zalsa_;
    use salsa::plumbing::interned as zalsa_struct_;

    impl PartialEq for SyntaxContextUnderlyingData {
        fn eq(&self, other: &Self) -> bool {
            self.outer_expn == other.outer_expn
                && self.outer_transparency == other.outer_transparency
                && self.edition == other.edition
                && self.parent == other.parent
        }
    }

    impl std::hash::Hash for SyntaxContextUnderlyingData {
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
        for SyntaxContextUnderlyingData
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
        const DEBUG_NAME: &'static str = "SyntaxContextData";
        type Fields<'a> = SyntaxContextUnderlyingData;
        type Struct<'a> = SyntaxContext;
        fn struct_from_id<'db>(id: salsa::Id) -> Self::Struct<'db> {
            SyntaxContext(id, std::marker::PhantomData)
        }
        fn deref_struct(s: Self::Struct<'_>) -> salsa::Id {
            s.0
        }
    }
    impl SyntaxContext {
        pub fn ingredient<Db>(db: &Db) -> &zalsa_struct_::IngredientImpl<Self>
        where
            Db: ?Sized + zalsa_::Database,
        {
            static CACHE: zalsa_::IngredientCache<zalsa_struct_::IngredientImpl<SyntaxContext>> =
                zalsa_::IngredientCache::new();
            CACHE.get_or_create(db.as_dyn_database(), || {
                db.zalsa().add_or_lookup_jar_by_type::<zalsa_struct_::JarImpl<SyntaxContext>>()
            })
        }
    }
    impl zalsa_::AsId for SyntaxContext {
        fn as_id(&self) -> salsa::Id {
            self.0
        }
    }
    impl zalsa_::FromId for SyntaxContext {
        fn from_id(id: salsa::Id) -> Self {
            Self(id, std::marker::PhantomData)
        }
    }
    unsafe impl Send for SyntaxContext {}

    unsafe impl Sync for SyntaxContext {}

    impl zalsa_::SalsaStructInDb for SyntaxContext {
        type MemoIngredientMap = salsa::plumbing::MemoIngredientSingletonIndex;

        fn lookup_or_create_ingredient_index(
            aux: &salsa::plumbing::Zalsa,
        ) -> salsa::plumbing::IngredientIndices {
            aux.add_or_lookup_jar_by_type::<zalsa_struct_::JarImpl<SyntaxContext>>().into()
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
                |id, data| SyntaxContextUnderlyingData {
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
            if self.is_root() {
                return None;
            }
            let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), self);
            std::clone::Clone::clone(&fields.outer_expn)
        }

        pub fn outer_transparency<Db>(self, db: &'db Db) -> Transparency
        where
            Db: ?Sized + zalsa_::Database,
        {
            if self.is_root() {
                return Transparency::Opaque;
            }
            let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), self);
            std::clone::Clone::clone(&fields.outer_transparency)
        }

        pub fn edition<Db>(self, db: &'db Db) -> Edition
        where
            Db: ?Sized + zalsa_::Database,
        {
            if self.is_root() {
                return Edition::from_u32(SyntaxContext::MAX_ID - self.0.as_u32());
            }
            let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), self);
            std::clone::Clone::clone(&fields.edition)
        }

        pub fn parent<Db>(self, db: &'db Db) -> SyntaxContext
        where
            Db: ?Sized + zalsa_::Database,
        {
            if self.is_root() {
                return self;
            }
            let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), self);
            std::clone::Clone::clone(&fields.parent)
        }

        /// This context, but with all transparent and semi-transparent expansions filtered away.
        pub fn opaque<Db>(self, db: &'db Db) -> SyntaxContext
        where
            Db: ?Sized + zalsa_::Database,
        {
            if self.is_root() {
                return self;
            }
            let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), self);
            std::clone::Clone::clone(&fields.opaque)
        }

        /// This context, but with all transparent expansions filtered away.
        pub fn opaque_and_semitransparent<Db>(self, db: &'db Db) -> SyntaxContext
        where
            Db: ?Sized + zalsa_::Database,
        {
            if self.is_root() {
                return self;
            }
            let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), self);
            std::clone::Clone::clone(&fields.opaque_and_semitransparent)
        }

        pub fn default_debug_fmt(this: Self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            salsa::with_attached_database(|db| {
                let fields = SyntaxContext::ingredient(db).fields(db.as_dyn_database(), this);
                let mut f = f.debug_struct("SyntaxContextData");
                let f = f.field("outer_expn", &fields.outer_expn);
                let f = f.field("outer_transparency", &fields.outer_expn);
                let f = f.field("edition", &fields.edition);
                let f = f.field("parent", &fields.parent);
                let f = f.field("opaque", &fields.opaque);
                let f = f.field("opaque_and_semitransparent", &fields.opaque_and_semitransparent);
                f.finish()
            })
            .unwrap_or_else(|| {
                f.debug_tuple("SyntaxContextData").field(&zalsa_::AsId::as_id(&this)).finish()
            })
        }
    }
};

impl SyntaxContext {
    const MAX_ID: u32 = salsa::Id::MAX_U32 - 1;

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
    pub const fn root(edition: Edition) -> Self {
        let edition = edition as u32;
        SyntaxContext(
            salsa::Id::from_u32(SyntaxContext::MAX_ID - edition),
            std::marker::PhantomData,
        )
    }

    pub fn into_u32(self) -> u32 {
        self.0.as_u32()
    }

    pub fn from_u32(u32: u32) -> Self {
        Self(salsa::Id::from_u32(u32), std::marker::PhantomData)
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
        write!(f, "{}", self.0.as_u32())
    }
}

impl std::fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "{}", self.0.as_u32())
        } else {
            f.debug_tuple("SyntaxContext").field(&self.0).finish()
        }
    }
}
