//! Defines [`EditionedFileId`], an interned wrapper around [`span::EditionedFileId`] that
//! is interned (so queries can take it) and remembers its crate.

use core::fmt;
use std::hash::{Hash, Hasher};

use span::Edition;
use vfs::FileId;

use crate::{Crate, RootQueryDb};

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EditionedFileId(
    salsa::Id,
    std::marker::PhantomData<&'static salsa::plumbing::interned::Value<EditionedFileId>>,
);

const _: () = {
    use salsa::plumbing as zalsa_;
    use zalsa_::interned as zalsa_struct_;
    type Configuration_ = EditionedFileId;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EditionedFileIdData {
        editioned_file_id: span::EditionedFileId,
        krate: Crate,
    }

    /// We like to include the origin crate in an `EditionedFileId` (for use in the item tree),
    /// but this poses us a problem.
    ///
    /// Spans contain `EditionedFileId`s, and we don't want to make them store the crate too
    /// because that will increase their size, which will increase memory usage significantly.
    /// Furthermore, things using spans do not generally need the crate: they are using the
    /// file id for queries like `ast_id_map` or `parse`, which do not care about the crate.
    ///
    /// To solve this, we hash **only the `span::EditionedFileId`**, but on still compare
    /// the crate in equality check. This preserves the invariant of `Hash` and `Eq` -
    /// although same hashes can be used for different items, same file ids used for multiple
    /// crates is a rare thing, and different items always have different hashes. Then,
    /// when we only have a `span::EditionedFileId`, we use the `intern()` method to
    /// reuse existing file ids, and create new one only if needed. See [`from_span_guess_origin`].
    ///
    /// See this for more info: https://rust-lang.zulipchat.com/#narrow/channel/185405-t-compiler.2Frust-analyzer/topic/Letting.20EditionedFileId.20know.20its.20crate/near/530189401
    ///
    /// [`from_span_guess_origin`]: EditionedFileId::from_span_guess_origin
    #[derive(Hash, PartialEq, Eq)]
    struct WithoutCrate {
        editioned_file_id: span::EditionedFileId,
    }

    impl Hash for EditionedFileIdData {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            let EditionedFileIdData { editioned_file_id, krate: _ } = *self;
            editioned_file_id.hash(state);
        }
    }

    impl zalsa_struct_::HashEqLike<WithoutCrate> for EditionedFileIdData {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            Hash::hash(self, state);
        }

        #[inline]
        fn eq(&self, data: &WithoutCrate) -> bool {
            let EditionedFileIdData { editioned_file_id, krate: _ } = *self;
            editioned_file_id == data.editioned_file_id
        }
    }

    impl zalsa_::HasJar for EditionedFileId {
        type Jar = zalsa_struct_::JarImpl<EditionedFileId>;
        const KIND: zalsa_::JarKind = zalsa_::JarKind::Struct;
    }

    zalsa_::register_jar! {
        zalsa_::ErasedJar::erase::<EditionedFileId>()
    }

    impl zalsa_struct_::Configuration for EditionedFileId {
        const LOCATION: salsa::plumbing::Location =
            salsa::plumbing::Location { file: file!(), line: line!() };
        const DEBUG_NAME: &'static str = "EditionedFileId";
        const REVISIONS: std::num::NonZeroUsize = std::num::NonZeroUsize::MAX;
        const PERSIST: bool = false;

        type Fields<'a> = EditionedFileIdData;
        type Struct<'db> = EditionedFileId;

        fn serialize<S>(_: &Self::Fields<'_>, _: S) -> Result<S::Ok, S::Error>
        where
            S: zalsa_::serde::Serializer,
        {
            unimplemented!("attempted to serialize value that set `PERSIST` to false")
        }

        fn deserialize<'de, D>(_: D) -> Result<Self::Fields<'static>, D::Error>
        where
            D: zalsa_::serde::Deserializer<'de>,
        {
            unimplemented!("attempted to deserialize value that cannot set `PERSIST` to false");
        }
    }

    impl Configuration_ {
        pub fn ingredient(zalsa: &zalsa_::Zalsa) -> &zalsa_struct_::IngredientImpl<Self> {
            static CACHE: zalsa_::IngredientCache<zalsa_struct_::IngredientImpl<EditionedFileId>> =
                zalsa_::IngredientCache::new();

            // SAFETY: `lookup_jar_by_type` returns a valid ingredient index, and the only
            // ingredient created by our jar is the struct ingredient.
            unsafe {
                CACHE.get_or_create(zalsa, || {
                    zalsa.lookup_jar_by_type::<zalsa_struct_::JarImpl<EditionedFileId>>()
                })
            }
        }
    }

    impl zalsa_::AsId for EditionedFileId {
        fn as_id(&self) -> salsa::Id {
            self.0.as_id()
        }
    }
    impl zalsa_::FromId for EditionedFileId {
        fn from_id(id: salsa::Id) -> Self {
            Self(<salsa::Id>::from_id(id), std::marker::PhantomData)
        }
    }

    unsafe impl Send for EditionedFileId {}
    unsafe impl Sync for EditionedFileId {}

    impl std::fmt::Debug for EditionedFileId {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Self::default_debug_fmt(*self, f)
        }
    }

    impl zalsa_::SalsaStructInDb for EditionedFileId {
        type MemoIngredientMap = salsa::plumbing::MemoIngredientSingletonIndex;

        fn lookup_ingredient_index(aux: &zalsa_::Zalsa) -> salsa::plumbing::IngredientIndices {
            aux.lookup_jar_by_type::<zalsa_struct_::JarImpl<EditionedFileId>>().into()
        }

        fn entries(zalsa: &zalsa_::Zalsa) -> impl Iterator<Item = zalsa_::DatabaseKeyIndex> + '_ {
            let _ingredient_index =
                zalsa.lookup_jar_by_type::<zalsa_struct_::JarImpl<EditionedFileId>>();
            <EditionedFileId>::ingredient(zalsa).entries(zalsa).map(|entry| entry.key())
        }

        #[inline]
        fn cast(id: salsa::Id, type_id: std::any::TypeId) -> Option<Self> {
            if type_id == std::any::TypeId::of::<EditionedFileId>() {
                Some(<Self as salsa::plumbing::FromId>::from_id(id))
            } else {
                None
            }
        }

        #[inline]
        unsafe fn memo_table(
            zalsa: &zalsa_::Zalsa,
            id: zalsa_::Id,
            current_revision: zalsa_::Revision,
        ) -> zalsa_::MemoTableWithTypes<'_> {
            // SAFETY: Guaranteed by caller.
            unsafe {
                zalsa.table().memos::<zalsa_struct_::Value<EditionedFileId>>(id, current_revision)
            }
        }
    }

    unsafe impl zalsa_::Update for EditionedFileId {
        unsafe fn maybe_update(old_pointer: *mut Self, new_value: Self) -> bool {
            if unsafe { *old_pointer } != new_value {
                unsafe { *old_pointer = new_value };
                true
            } else {
                false
            }
        }
    }

    impl EditionedFileId {
        pub fn from_span(
            db: &(impl salsa::Database + ?Sized),
            editioned_file_id: span::EditionedFileId,
            krate: Crate,
        ) -> Self {
            let (zalsa, zalsa_local) = db.zalsas();
            Configuration_::ingredient(zalsa).intern(
                zalsa,
                zalsa_local,
                EditionedFileIdData { editioned_file_id, krate },
                |_, data| data,
            )
        }

        /// Guesses the crate for the file.
        ///
        /// Only use this if you cannot precisely determine the origin. This can happen in one of two cases:
        ///
        ///  1. The file is not in the module tree.
        ///  2. You are latency sensitive and cannot afford calling the def map to precisely compute the origin
        ///     (e.g. on enter feature, folding, etc.).
        pub fn from_span_guess_origin(
            db: &dyn RootQueryDb,
            editioned_file_id: span::EditionedFileId,
        ) -> Self {
            let (zalsa, zalsa_local) = db.zalsas();
            Configuration_::ingredient(zalsa).intern(
                zalsa,
                zalsa_local,
                WithoutCrate { editioned_file_id },
                |_, _| {
                    // FileId not in the database.
                    let krate = db
                        .relevant_crates(editioned_file_id.file_id())
                        .first()
                        .copied()
                        .or_else(|| db.all_crates().first().copied())
                        .unwrap_or_else(|| {
                            // What we're doing here is a bit fishy. We rely on the fact that we only need
                            // the crate in the item tree, and we should not create an `EditionedFileId`
                            // without a crate except in cases where it does not matter. The chances that
                            // `all_crates()` will be empty are also very slim, but it can occur during startup.
                            // In the very unlikely case that there is a bug and we'll use this crate, Salsa
                            // will panic.

                            // SAFETY: 0 is less than `Id::MAX_U32`.
                            salsa::plumbing::FromId::from_id(unsafe { salsa::Id::from_index(0) })
                        });
                    EditionedFileIdData { editioned_file_id, krate }
                },
            )
        }

        pub fn editioned_file_id(self, db: &dyn salsa::Database) -> span::EditionedFileId {
            let zalsa = db.zalsa();
            let fields = Configuration_::ingredient(zalsa).fields(zalsa, self);
            fields.editioned_file_id
        }

        pub fn krate(self, db: &dyn salsa::Database) -> Crate {
            let zalsa = db.zalsa();
            let fields = Configuration_::ingredient(zalsa).fields(zalsa, self);
            fields.krate
        }

        /// Default debug formatting for this struct (may be useful if you define your own `Debug` impl)
        pub fn default_debug_fmt(this: Self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            zalsa_::with_attached_database(|db| {
                let zalsa = db.zalsa();
                let fields = Configuration_::ingredient(zalsa).fields(zalsa, this);
                fmt::Debug::fmt(fields, f)
            })
            .unwrap_or_else(|| {
                f.debug_tuple("EditionedFileId").field(&zalsa_::AsId::as_id(&this)).finish()
            })
        }
    }
};

impl EditionedFileId {
    #[inline]
    pub fn new(db: &dyn salsa::Database, file_id: FileId, edition: Edition, krate: Crate) -> Self {
        EditionedFileId::from_span(db, span::EditionedFileId::new(file_id, edition), krate)
    }

    /// Attaches the current edition and guesses the crate for the file.
    ///
    /// Only use this if you cannot precisely determine the origin. This can happen in one of two cases:
    ///
    ///  1. The file is not in the module tree.
    ///  2. You are latency sensitive and cannot afford calling the def map to precisely compute the origin
    ///     (e.g. on enter feature, folding, etc.).
    #[inline]
    pub fn current_edition_guess_origin(db: &dyn RootQueryDb, file_id: FileId) -> Self {
        Self::from_span_guess_origin(db, span::EditionedFileId::current_edition(file_id))
    }

    #[inline]
    pub fn file_id(self, db: &dyn salsa::Database) -> vfs::FileId {
        let id = self.editioned_file_id(db);
        id.file_id()
    }

    #[inline]
    pub fn unpack(self, db: &dyn salsa::Database) -> (vfs::FileId, span::Edition) {
        let id = self.editioned_file_id(db);
        (id.file_id(), id.edition())
    }

    #[inline]
    pub fn edition(self, db: &dyn salsa::Database) -> Edition {
        self.editioned_file_id(db).edition()
    }
}
