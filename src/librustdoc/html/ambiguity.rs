#![allow(warnings)]
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_span::Symbol;

use crate::clean::{self, FnDecl, Function};

#[derive(Debug, Clone)]
pub struct PathLike<'a> {
    path: &'a clean::Path,
    suffix_len: usize,
}

impl<'a> PathLike<'a> {
    pub fn make_longer(self) -> Self {
        let Self { path, suffix_len } = self;
        assert!(suffix_len < path.segments.len());
        Self { path, suffix_len: suffix_len + 1 }
    }

    pub fn iter_from_end(&self) -> impl DoubleEndedIterator<Item = &'a Symbol> {
        self.path.segments.iter().map(|seg| &seg.name).rev().take(self.suffix_len)
    }
}

impl<'a> Eq for PathLike<'a> {}
impl<'a> PartialEq for PathLike<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.suffix_len != other.suffix_len {
            return false;
        }
        let self_suffix = self.iter_from_end();
        let other_suffix = self.iter_from_end();
        self_suffix.zip(other_suffix).all(|(s, o)| s.eq(o))
    }
}

impl<'a> std::hash::Hash for PathLike<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for sym in self.iter_from_end() {
            sym.hash(state)
        }
    }
}

#[derive(Debug, Clone)]
pub struct AmbiguityTable<'a> {
    inner: FxHashMap<DefId, PathLike<'a>>,
}

impl<'a> AmbiguityTable<'a> {
    pub fn empty() -> Self {
        Self { inner: FxHashMap::default() }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn build() -> AmbiguityTableBuilder<'a> {
        AmbiguityTableBuilder { inner: FxHashMap::default() }
    }

    pub fn get(&self, x: &DefId) -> Option<&PathLike<'a>> {
        self.inner.get(x)
    }
}

enum AmbiguityTableBuilderEntry {
    MapsToOne(DefId),
    IsAmbiguous,
}

pub struct AmbiguityTableBuilder<'a> {
    inner: FxHashMap<PathLike<'a>, AmbiguityTableBuilderEntry>,
}

impl<'a> AmbiguityTableBuilder<'a> {
    // Invariant: must start with length 1 path view
    fn add_path_view(&mut self, p: PathLike<'a>, did: DefId) {
        use std::collections::hash_map::Entry::*;
        // TODO: clone
        match self.inner.entry(p.clone()) {
            Occupied(entry) => {
                let v =
                    std::mem::replace(entry.into_mut(), AmbiguityTableBuilderEntry::IsAmbiguous);
                let p = p.make_longer();
                match v {
                    AmbiguityTableBuilderEntry::MapsToOne(other_did) => {
                        self.add_path_view(p.clone(), other_did)
                    }
                    AmbiguityTableBuilderEntry::IsAmbiguous => (),
                }
                self.add_path_view(p.clone(), did)
            }
            Vacant(entry) => {
                entry.insert(AmbiguityTableBuilderEntry::MapsToOne(did));
            }
        }
    }

    fn add_path(&mut self, path: &'a clean::Path) {
        let pv = PathLike { path, suffix_len: 1 };
        self.add_path_view(pv, path.def_id())
    }

    fn add_generic_bound(&mut self, generic_bound: &'a clean::GenericBound) {
        match generic_bound {
            clean::GenericBound::TraitBound(poly_trait, _) => self.add_poly_trait(poly_trait),
            clean::GenericBound::Outlives(_) => (),
        }
    }

    fn add_poly_trait(&mut self, poly_trait: &'a clean::PolyTrait) {
        // TODO: also add potential bounds on trait parameters
        self.add_path(&poly_trait.trait_);
        for gen_param in &poly_trait.generic_params {
            use clean::GenericParamDefKind::*;
            match &gen_param.kind {
                Type { bounds, .. } => {
                    for bnd in bounds {
                        self.add_generic_bound(bnd)
                    }
                }
                Lifetime { .. } | Const { .. } => (),
            }
        }
    }

    fn add_type(&mut self, ty: &'a clean::Type) {
        match ty {
            clean::Type::Path { path } => self.add_path(path),
            clean::Type::Tuple(tys) => {
                for ty in tys {
                    self.add_type(ty)
                }
            }
            clean::Type::RawPointer(_, ty)
            | clean::Type::Slice(ty)
            | clean::Type::BorrowedRef { type_: ty, .. }
            | clean::Type::Array(ty, _) => self.add_type(ty),

            clean::Type::DynTrait(poly_trait, _) => {
                for trai in poly_trait {
                    self.add_poly_trait(trai)
                }
            }

            clean::Type::BareFunction(bare_func_decl) => {
                let clean::FnDecl { output, inputs, .. } = &bare_func_decl.decl;
                self.add_type(output);
                for inpt in &inputs.values {
                    self.add_type(&inpt.type_)
                }
            }
            clean::Type::ImplTrait(bnds) => {
                for bnd in bnds {
                    self.add_generic_bound(bnd)
                }
            }
            clean::Type::Infer
            | clean::Type::QPath(_)
            | clean::Type::Primitive(_)
            | clean::Type::Generic(_) => (),
        }
    }

    fn add_fndecl(&mut self, decl: &'a FnDecl) {
        for arg in &decl.inputs.values {
            self.add_type(&arg.type_);
        }
        self.add_type(&decl.output);
    }

    pub fn add_fn(mut self, f: &'a Function) -> Self{
        self.add_fndecl(&f.decl);
        self
    }

    pub fn finnish(self) -> AmbiguityTable<'a> {
        let mut inner = FxHashMap::default();
        for (path_view, did) in self.inner {
            if let AmbiguityTableBuilderEntry::MapsToOne(did) = did {
                let hopefully_none = inner.insert(did, path_view);
                assert!(hopefully_none.is_none());
            }
        }
        AmbiguityTable { inner }
    }
}
