use crate::ty::{self, TyCtxt};
use crate::hir::map::definitions::FIRST_FREE_DEF_INDEX;
use rustc_data_structures::indexed_vec::Idx;
use serialize;
use std::fmt;
use std::u32;

newtype_index! {
    pub struct CrateId {
        ENCODABLE = custom
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrateNum {
    /// Virtual crate for builtin macros
    // FIXME(jseyfried): this is also used for custom derives until proc-macro crates get
    // `CrateNum`s.
    BuiltinMacros,
    /// A special CrateNum that we use for the tcx.rcache when decoding from
    /// the incr. comp. cache.
    ReservedForIncrCompCache,
    Index(CrateId),
}

impl ::std::fmt::Debug for CrateNum {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        match self {
            CrateNum::Index(id) => write!(fmt, "crate{}", id.private),
            CrateNum::BuiltinMacros => write!(fmt, "builtin macros crate"),
            CrateNum::ReservedForIncrCompCache => write!(fmt, "crate for decoding incr comp cache"),
        }
    }
}

/// Item definitions in the currently-compiled crate would have the CrateNum
/// LOCAL_CRATE in their DefId.
pub const LOCAL_CRATE: CrateNum = CrateNum::Index(CrateId::from_u32_const(0));


impl Idx for CrateNum {
    #[inline]
    fn new(value: usize) -> Self {
        CrateNum::Index(Idx::new(value))
    }

    #[inline]
    fn index(self) -> usize {
        match self {
            CrateNum::Index(idx) => Idx::index(idx),
            _ => bug!("Tried to get crate index of {:?}", self),
        }
    }
}

impl CrateNum {
    pub fn new(x: usize) -> CrateNum {
        CrateNum::from_usize(x)
    }

    pub fn from_usize(x: usize) -> CrateNum {
        CrateNum::Index(CrateId::from_usize(x))
    }

    pub fn from_u32(x: u32) -> CrateNum {
        CrateNum::Index(CrateId::from_u32(x))
    }

    pub fn as_usize(self) -> usize {
        match self {
            CrateNum::Index(id) => id.as_usize(),
            _ => bug!("tried to get index of nonstandard crate {:?}", self),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            CrateNum::Index(id) => id.as_u32(),
            _ => bug!("tried to get index of nonstandard crate {:?}", self),
        }
    }

    pub fn as_def_id(&self) -> DefId { DefId { krate: *self, index: CRATE_DEF_INDEX } }
}

impl fmt::Display for CrateNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrateNum::Index(id) => fmt::Display::fmt(&id.private, f),
            CrateNum::BuiltinMacros => write!(f, "builtin macros crate"),
            CrateNum::ReservedForIncrCompCache => write!(f, "crate for decoding incr comp cache"),
        }
    }
}

impl serialize::UseSpecializedEncodable for CrateNum {}
impl serialize::UseSpecializedDecodable for CrateNum {}

newtype_index! {
    /// A DefIndex is an index into the hir-map for a crate, identifying a
    /// particular definition. It should really be considered an interned
    /// shorthand for a particular DefPath.
    pub struct DefIndex {
        DEBUG_FORMAT = "DefIndex({})",

        /// The crate root is always assigned index 0 by the AST Map code,
        /// thanks to `NodeCollector::new`.
        const CRATE_DEF_INDEX = 0,
    }
}

impl DefIndex {
    // Proc macros from a proc-macro crate have a kind of virtual DefIndex. This
    // function maps the index of the macro within the crate (which is also the
    // index of the macro in the CrateMetadata::proc_macros array) to the
    // corresponding DefIndex.
    pub fn from_proc_macro_index(proc_macro_index: usize) -> DefIndex {
        // DefIndex for proc macros start from FIRST_FREE_DEF_INDEX,
        // because the first FIRST_FREE_DEF_INDEX indexes are reserved
        // for internal use.
        let def_index = DefIndex::from(
            proc_macro_index.checked_add(FIRST_FREE_DEF_INDEX)
                .expect("integer overflow adding `proc_macro_index`"));
        assert!(def_index != CRATE_DEF_INDEX);
        def_index
    }

    // This function is the reverse of from_proc_macro_index() above.
    pub fn to_proc_macro_index(self: DefIndex) -> usize {
        self.index().checked_sub(FIRST_FREE_DEF_INDEX)
            .unwrap_or_else(|| {
                bug!("using local index {:?} as proc-macro index", self)
            })
    }
}

impl serialize::UseSpecializedEncodable for DefIndex {}
impl serialize::UseSpecializedDecodable for DefIndex {}

/// A `DefId` identifies a particular *definition*, by combining a crate
/// index and a def index.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub index: DefIndex,
}

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DefId({}:{}", self.krate, self.index.index())?;

        ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                write!(f, " ~ {}", tcx.def_path_debug_str(*self))?;
            }
            Ok(())
        })?;

        write!(f, ")")
    }
}

impl DefId {
    /// Makes a local `DefId` from the given `DefIndex`.
    #[inline]
    pub fn local(index: DefIndex) -> DefId {
        DefId { krate: LOCAL_CRATE, index: index }
    }

    #[inline]
    pub fn is_local(self) -> bool {
        self.krate == LOCAL_CRATE
    }

    #[inline]
    pub fn to_local(self) -> LocalDefId {
        LocalDefId::from_def_id(self)
    }

    pub fn describe_as_module(&self, tcx: TyCtxt<'_>) -> String {
        if self.is_local() && self.index == CRATE_DEF_INDEX {
            format!("top-level module")
        } else {
            format!("module `{}`", tcx.def_path_str(*self))
        }
    }
}

impl serialize::UseSpecializedEncodable for DefId {}
impl serialize::UseSpecializedDecodable for DefId {}

/// A LocalDefId is equivalent to a DefId with `krate == LOCAL_CRATE`. Since
/// we encode this information in the type, we can ensure at compile time that
/// no DefIds from upstream crates get thrown into the mix. There are quite a
/// few cases where we know that only DefIds from the local crate are expected
/// and a DefId from a different crate would signify a bug somewhere. This
/// is when LocalDefId comes in handy.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalDefId(DefIndex);

impl LocalDefId {
    #[inline]
    pub fn from_def_id(def_id: DefId) -> LocalDefId {
        assert!(def_id.is_local());
        LocalDefId(def_id.index)
    }

    #[inline]
    pub fn to_def_id(self) -> DefId {
        DefId {
            krate: LOCAL_CRATE,
            index: self.0
        }
    }
}

impl fmt::Debug for LocalDefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_def_id().fmt(f)
    }
}

impl serialize::UseSpecializedEncodable for LocalDefId {}
impl serialize::UseSpecializedDecodable for LocalDefId {}
