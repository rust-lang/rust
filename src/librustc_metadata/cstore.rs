// The crate store - a central repo for information collected about external
// crates and libraries

use crate::schema;
use rustc::hir::def_id::{CrateNum, DefIndex};
use rustc::hir::map::definitions::DefPathTable;
use rustc::middle::cstore::{DepKind, ExternCrate, MetadataLoader};
use rustc::mir::interpret::AllocDecodingState;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::util::nodemap::{FxHashMap, NodeMap};

use rustc_data_structures::sync::{Lrc, RwLock, Lock};
use syntax::ast;
use syntax::ext::base::SyntaxExtension;
use syntax::symbol::Symbol;
use syntax_pos;

pub use rustc::middle::cstore::{NativeLibrary, NativeLibraryKind, LinkagePreference};
pub use rustc::middle::cstore::NativeLibraryKind::*;
pub use rustc::middle::cstore::{CrateSource, LibSource, ForeignModule};

pub use crate::cstore_impl::{provide, provide_extern};

// A map from external crate numbers (as decoded from some crate file) to
// local crate numbers (as generated during this session). Each external
// crate may refer to types in other external crates, and each has their
// own crate numbers.
pub type CrateNumMap = IndexVec<CrateNum, CrateNum>;

pub use rustc_data_structures::sync::MetadataRef;

pub struct MetadataBlob(pub MetadataRef);

/// Holds information about a syntax_pos::SourceFile imported from another crate.
/// See `imported_source_files()` for more information.
pub struct ImportedSourceFile {
    /// This SourceFile's byte-offset within the source_map of its original crate
    pub original_start_pos: syntax_pos::BytePos,
    /// The end of this SourceFile within the source_map of its original crate
    pub original_end_pos: syntax_pos::BytePos,
    /// The imported SourceFile's representation within the local source_map
    pub translated_source_file: Lrc<syntax_pos::SourceFile>,
}

pub struct CrateMetadata {
    /// Original name of the crate.
    pub name: Symbol,

    /// Name of the crate as imported. I.e., if imported with
    /// `extern crate foo as bar;` this will be `bar`.
    pub imported_name: Symbol,

    /// Information about the extern crate that caused this crate to
    /// be loaded. If this is `None`, then the crate was injected
    /// (e.g., by the allocator)
    pub extern_crate: Lock<Option<ExternCrate>>,

    pub blob: MetadataBlob,
    pub cnum_map: CrateNumMap,
    pub cnum: CrateNum,
    pub dependencies: Lock<Vec<CrateNum>>,
    pub source_map_import_info: RwLock<Vec<ImportedSourceFile>>,

    /// Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    pub alloc_decoding_state: AllocDecodingState,

    // NOTE(eddyb) we pass `'static` to a `'tcx` parameter because this
    // lifetime is only used behind `Lazy` / `LazySeq`, and therefore
    // acts like an universal (`for<'tcx>`), that is paired up with
    // whichever `TyCtxt` is being used to decode those values.
    pub root: schema::CrateRoot<'static>,

    /// For each public item in this crate, we encode a key. When the
    /// crate is loaded, we read all the keys and put them in this
    /// hashmap, which gives the reverse mapping. This allows us to
    /// quickly retrace a `DefPath`, which is needed for incremental
    /// compilation support.
    pub def_path_table: Lrc<DefPathTable>,

    pub trait_impls: FxHashMap<(u32, DefIndex), schema::LazySeq<DefIndex>>,

    pub dep_kind: Lock<DepKind>,
    pub source: CrateSource,

    pub proc_macros: Option<Vec<(ast::Name, Lrc<SyntaxExtension>)>>,

    /// Whether or not this crate should be consider a private dependency
    /// for purposes of the 'exported_private_dependencies' lint
    pub private_dep: bool
}

pub struct CStore {
    metas: RwLock<IndexVec<CrateNum, Option<Lrc<CrateMetadata>>>>,
    /// Map from NodeId's of local extern crate statements to crate numbers
    extern_mod_crate_map: Lock<NodeMap<CrateNum>>,
    pub metadata_loader: Box<dyn MetadataLoader + Sync>,
}

pub enum LoadedMacro {
    MacroDef(ast::Item),
    ProcMacro(Lrc<SyntaxExtension>),
}

impl CStore {
    pub fn new(metadata_loader: Box<dyn MetadataLoader + Sync>) -> CStore {
        CStore {
            // We add an empty entry for LOCAL_CRATE (which maps to zero) in
            // order to make array indices in `metas` match with the
            // corresponding `CrateNum`. This first entry will always remain
            // `None`.
            metas: RwLock::new(IndexVec::from_elem_n(None, 1)),
            extern_mod_crate_map: Default::default(),
            metadata_loader,
        }
    }

    pub(super) fn alloc_new_crate_num(&self) -> CrateNum {
        let mut metas = self.metas.borrow_mut();
        let cnum = CrateNum::new(metas.len());
        metas.push(None);
        cnum
    }

    pub(super) fn get_crate_data(&self, cnum: CrateNum) -> Lrc<CrateMetadata> {
        self.metas.borrow()[cnum].clone()
            .unwrap_or_else(|| panic!("Failed to get crate data for {:?}", cnum))
    }

    pub(super) fn set_crate_data(&self, cnum: CrateNum, data: Lrc<CrateMetadata>) {
        let mut metas = self.metas.borrow_mut();
        assert!(metas[cnum].is_none(), "Overwriting crate metadata entry");
        metas[cnum] = Some(data);
    }

    pub(super) fn iter_crate_data<I>(&self, mut i: I)
        where I: FnMut(CrateNum, &Lrc<CrateMetadata>)
    {
        for (k, v) in self.metas.borrow().iter_enumerated() {
            if let &Some(ref v) = v {
                i(k, v);
            }
        }
    }

    pub(super) fn crate_dependencies_in_rpo(&self, krate: CrateNum) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        self.push_dependencies_in_postorder(&mut ordering, krate);
        ordering.reverse();
        ordering
    }

    pub(super) fn push_dependencies_in_postorder(&self,
                                                 ordering: &mut Vec<CrateNum>,
                                                 krate: CrateNum) {
        if ordering.contains(&krate) {
            return;
        }

        let data = self.get_crate_data(krate);
        for &dep in data.dependencies.borrow().iter() {
            if dep != krate {
                self.push_dependencies_in_postorder(ordering, dep);
            }
        }

        ordering.push(krate);
    }

    pub(super) fn do_postorder_cnums_untracked(&self) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        for (num, v) in self.metas.borrow().iter_enumerated() {
            if let &Some(_) = v {
                self.push_dependencies_in_postorder(&mut ordering, num);
            }
        }
        return ordering
    }

    pub(super) fn add_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId, cnum: CrateNum) {
        self.extern_mod_crate_map.borrow_mut().insert(emod_id, cnum);
    }

    pub(super) fn do_extern_mod_stmt_cnum(&self, emod_id: ast::NodeId) -> Option<CrateNum> {
        self.extern_mod_crate_map.borrow().get(&emod_id).cloned()
    }
}
