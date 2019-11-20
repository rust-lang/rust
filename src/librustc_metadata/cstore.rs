// The crate store - a central repo for information collected about external
// crates and libraries

use crate::rmeta::CrateMetadata;

use rustc_data_structures::sync::Lrc;
use rustc_index::vec::IndexVec;
use rustc::hir::def_id::CrateNum;
use syntax::ast;
use syntax::edition::Edition;
use syntax::expand::allocator::AllocatorKind;
use syntax_expand::base::SyntaxExtension;

pub use crate::rmeta::{provide, provide_extern};

#[derive(Clone)]
pub struct CStore {
    metas: IndexVec<CrateNum, Option<Lrc<CrateMetadata>>>,
    crate injected_panic_runtime: Option<CrateNum>,
    crate allocator_kind: Option<AllocatorKind>,
}

pub enum LoadedMacro {
    MacroDef(ast::Item, Edition),
    ProcMacro(SyntaxExtension),
}

impl Default for CStore {
    fn default() -> Self {
        CStore {
            // We add an empty entry for LOCAL_CRATE (which maps to zero) in
            // order to make array indices in `metas` match with the
            // corresponding `CrateNum`. This first entry will always remain
            // `None`.
            metas: IndexVec::from_elem_n(None, 1),
            injected_panic_runtime: None,
            allocator_kind: None,
        }
    }
}

impl CStore {
    crate fn alloc_new_crate_num(&mut self) -> CrateNum {
        self.metas.push(None);
        CrateNum::new(self.metas.len() - 1)
    }

    crate fn get_crate_data(&self, cnum: CrateNum) -> &CrateMetadata {
        self.metas[cnum].as_ref()
            .unwrap_or_else(|| panic!("Failed to get crate data for {:?}", cnum))
    }

    crate fn set_crate_data(&mut self, cnum: CrateNum, data: CrateMetadata) {
        assert!(self.metas[cnum].is_none(), "Overwriting crate metadata entry");
        self.metas[cnum] = Some(Lrc::new(data));
    }

    crate fn iter_crate_data<I>(&self, mut i: I)
        where I: FnMut(CrateNum, &CrateMetadata)
    {
        for (k, v) in self.metas.iter_enumerated() {
            if let &Some(ref v) = v {
                i(k, v);
            }
        }
    }

    crate fn crate_dependencies_in_rpo(&self, krate: CrateNum) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        self.push_dependencies_in_postorder(&mut ordering, krate);
        ordering.reverse();
        ordering
    }

    crate fn push_dependencies_in_postorder(&self, ordering: &mut Vec<CrateNum>, krate: CrateNum) {
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

    crate fn do_postorder_cnums_untracked(&self) -> Vec<CrateNum> {
        let mut ordering = Vec::new();
        for (num, v) in self.metas.iter_enumerated() {
            if let &Some(_) = v {
                self.push_dependencies_in_postorder(&mut ordering, num);
            }
        }
        return ordering
    }
}
