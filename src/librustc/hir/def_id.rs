use crate::ty::{self, print::Printer, subst::Kind, Ty, TyCtxt};
use crate::hir::map::definitions::{DisambiguatedDefPathData, FIRST_FREE_HIGH_DEF_INDEX};
use rustc_data_structures::indexed_vec::Idx;
use serialize;
use std::fmt;
use std::u32;
use syntax::symbol::{LocalInternedString, Symbol};

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

/// A DefIndex is an index into the hir-map for a crate, identifying a
/// particular definition. It should really be considered an interned
/// shorthand for a particular DefPath.
///
/// At the moment we are allocating the numerical values of DefIndexes from two
/// address spaces: DefIndexAddressSpace::Low and DefIndexAddressSpace::High.
/// This allows us to allocate the DefIndexes of all item-likes
/// (Items, TraitItems, and ImplItems) into one of these spaces and
/// consequently use a simple array for lookup tables keyed by DefIndex and
/// known to be densely populated. This is especially important for the HIR map.
///
/// Since the DefIndex is mostly treated as an opaque ID, you probably
/// don't have to care about these address spaces.

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub struct DefIndex(u32);

/// The crate root is always assigned index 0 by the AST Map code,
/// thanks to `NodeCollector::new`.
pub const CRATE_DEF_INDEX: DefIndex = DefIndex(0);

impl fmt::Debug for DefIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,
               "DefIndex({}:{})",
               self.address_space().index(),
               self.as_array_index())
    }
}

impl DefIndex {
    #[inline]
    pub fn address_space(&self) -> DefIndexAddressSpace {
        match self.0 & 1 {
            0 => DefIndexAddressSpace::Low,
            1 => DefIndexAddressSpace::High,
            _ => unreachable!()
        }
    }

    /// Converts this DefIndex into a zero-based array index.
    /// This index is the offset within the given DefIndexAddressSpace.
    #[inline]
    pub fn as_array_index(&self) -> usize {
        (self.0 >> 1) as usize
    }

    #[inline]
    pub fn from_array_index(i: usize, address_space: DefIndexAddressSpace) -> DefIndex {
        DefIndex::from_raw_u32(((i << 1) | (address_space as usize)) as u32)
    }

    // Proc macros from a proc-macro crate have a kind of virtual DefIndex. This
    // function maps the index of the macro within the crate (which is also the
    // index of the macro in the CrateMetadata::proc_macros array) to the
    // corresponding DefIndex.
    pub fn from_proc_macro_index(proc_macro_index: usize) -> DefIndex {
        // DefIndex for proc macros start from FIRST_FREE_HIGH_DEF_INDEX,
        // because the first FIRST_FREE_HIGH_DEF_INDEX indexes are reserved
        // for internal use.
        let def_index = DefIndex::from_array_index(
            proc_macro_index.checked_add(FIRST_FREE_HIGH_DEF_INDEX)
                .expect("integer overflow adding `proc_macro_index`"),
            DefIndexAddressSpace::High);
        assert!(def_index != CRATE_DEF_INDEX);
        def_index
    }

    // This function is the reverse of from_proc_macro_index() above.
    pub fn to_proc_macro_index(self: DefIndex) -> usize {
        assert_eq!(self.address_space(), DefIndexAddressSpace::High);

        self.as_array_index().checked_sub(FIRST_FREE_HIGH_DEF_INDEX)
            .unwrap_or_else(|| {
                bug!("using local index {:?} as proc-macro index", self)
            })
    }

    // Don't use this if you don't know about the DefIndex encoding.
    pub fn from_raw_u32(x: u32) -> DefIndex {
        DefIndex(x)
    }

    // Don't use this if you don't know about the DefIndex encoding.
    pub fn as_raw_u32(&self) -> u32 {
        self.0
    }
}

impl serialize::UseSpecializedEncodable for DefIndex {}
impl serialize::UseSpecializedDecodable for DefIndex {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DefIndexAddressSpace {
    Low = 0,
    High = 1,
}

impl DefIndexAddressSpace {
    #[inline]
    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// A `DefId` identifies a particular *definition*, by combining a crate
/// index and a def index.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub index: DefIndex,
}

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DefId({}/{}:{}",
               self.krate,
               self.index.address_space().index(),
               self.index.as_array_index())?;

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

    pub fn describe_as_module(&self, tcx: TyCtxt<'_, '_, '_>) -> String {
        if self.is_local() && self.index == CRATE_DEF_INDEX {
            format!("top-level module")
        } else {
            format!("module `{}`", tcx.def_path_str(*self))
        }
    }

    /// Check if a `DefId`'s path matches the given absolute type path usage.
    // Uplifted from rust-lang/rust-clippy
    pub fn match_path<'a, 'tcx>(self, tcx: TyCtxt<'a, 'tcx, 'tcx>, path: &[&str]) -> bool {
        pub struct AbsolutePathPrinter<'a, 'tcx> {
            pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
        }

        impl<'tcx> Printer<'tcx, 'tcx> for AbsolutePathPrinter<'_, 'tcx> {
            type Error = !;

            type Path = Vec<LocalInternedString>;
            type Region = ();
            type Type = ();
            type DynExistential = ();

            fn tcx<'a>(&'a self) -> TyCtxt<'a, 'tcx, 'tcx> {
                self.tcx
            }

            fn print_region(self, _region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
                Ok(())
            }

            fn print_type(self, _ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
                Ok(())
            }

            fn print_dyn_existential(
                self,
                _predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
                ) -> Result<Self::DynExistential, Self::Error> {
                Ok(())
            }

            fn path_crate(self, cnum: CrateNum) -> Result<Self::Path, Self::Error> {
                Ok(vec![self.tcx.original_crate_name(cnum).as_str()])
            }

            fn path_qualified(
                self,
                self_ty: Ty<'tcx>,
                trait_ref: Option<ty::TraitRef<'tcx>>,
                ) -> Result<Self::Path, Self::Error> {
                if trait_ref.is_none() {
                    if let ty::Adt(def, substs) = self_ty.sty {
                        return self.print_def_path(def.did, substs);
                    }
                }

                // This shouldn't ever be needed, but just in case:
                Ok(vec![match trait_ref {
                    Some(trait_ref) => Symbol::intern(&format!("{:?}", trait_ref)).as_str(),
                    None => Symbol::intern(&format!("<{}>", self_ty)).as_str(),
                }])
            }

            fn path_append_impl(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _disambiguated_data: &DisambiguatedDefPathData,
                self_ty: Ty<'tcx>,
                trait_ref: Option<ty::TraitRef<'tcx>>,
                ) -> Result<Self::Path, Self::Error> {
                let mut path = print_prefix(self)?;

                // This shouldn't ever be needed, but just in case:
                path.push(match trait_ref {
                    Some(trait_ref) => {
                        Symbol::intern(&format!("<impl {} for {}>", trait_ref, self_ty)).as_str()
                    },
                    None => Symbol::intern(&format!("<impl {}>", self_ty)).as_str(),
                });

                Ok(path)
            }

            fn path_append(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                disambiguated_data: &DisambiguatedDefPathData,
                ) -> Result<Self::Path, Self::Error> {
                let mut path = print_prefix(self)?;
                path.push(disambiguated_data.data.as_interned_str().as_str());
                Ok(path)
            }

            fn path_generic_args(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _args: &[Kind<'tcx>],
                ) -> Result<Self::Path, Self::Error> {
                print_prefix(self)
            }
        }

        let names = AbsolutePathPrinter { tcx }.print_def_path(self, &[]).unwrap();

        names.len() == path.len()
            && names.into_iter().zip(path.iter()).all(|(a, &b)| *a == *b)
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
