// The crate store - a central repo for information collected about external
// crates and libraries

use crate::rmeta::CrateMetadata;

use rustc_data_structures::sync::Lrc;
use rustc_index::vec::IndexVec;
use rustc::hir::def_id::{LOCAL_CRATE, CrateNum};
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

    crate fn iter_crate_data(&self, mut f: impl FnMut(CrateNum, &CrateMetadata)) {
        for (cnum, data) in self.metas.iter_enumerated() {
            if let Some(data) = data {
                f(cnum, data);
            }
        }
    }

    fn push_dependencies_in_postorder(&self, deps: &mut Vec<CrateNum>, cnum: CrateNum) {
        if !deps.contains(&cnum) {
            let data = self.get_crate_data(cnum);
            for &dep in data.dependencies().iter() {
                if dep != cnum {
                    self.push_dependencies_in_postorder(deps, dep);
                }
            }

            deps.push(cnum);
        }
    }

    crate fn crate_dependencies_in_postorder(&self, cnum: CrateNum) -> Vec<CrateNum> {
        let mut deps = Vec::new();
        if cnum == LOCAL_CRATE {
            self.iter_crate_data(|cnum, _| self.push_dependencies_in_postorder(&mut deps, cnum));
        } else {
            self.push_dependencies_in_postorder(&mut deps, cnum);
        }
        deps
    }

    crate fn crate_dependencies_in_reverse_postorder(&self, cnum: CrateNum) -> Vec<CrateNum> {
        let mut deps = self.crate_dependencies_in_postorder(cnum);
        deps.reverse();
        deps
    }
}
