use std::{cmp::Ordering, hash::Hash};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_query_system::ich::StableHashingContext;

use crate::ty;

use super::{AllocationModuloRelocations, ConstAlloc, ConstValue, GlobalAlloc, Scalar};

impl<'a> HashStable<StableHashingContext<'a>> for ConstAlloc<'_> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.ty.hash_stable(hcx, hasher);
        ty::tls::with_opt(|tcx| {
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            tcx.get_global_alloc(self.alloc_id).hash_stable(hcx, hasher);
        });
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for ConstValue<'_> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        ty::tls::with_opt(|tcx| match self {
            ConstValue::Scalar(scalar) => {
                std::mem::discriminant(scalar).hash_stable(hcx, hasher);
                match scalar {
                    Scalar::Int(int) => int.hash_stable(hcx, hasher),
                    Scalar::Ptr(ptr, size) => {
                        let tcx = tcx.expect("can't hash AllocIds during hir lowering");
                        let (ptr, offset) = ptr.into_parts();
                        tcx.get_global_alloc(ptr).hash_stable(hcx, hasher);
                        offset.hash_stable(hcx, hasher);
                        size.hash_stable(hcx, hasher);
                    }
                }
            }
            ConstValue::Slice { data, start, end } => {
                AllocationModuloRelocations(*data).hash_stable(hcx, hasher);
                start.hash_stable(hcx, hasher);
                end.hash_stable(hcx, hasher);
            }
            ConstValue::ByRef { alloc, offset } => {
                AllocationModuloRelocations(*alloc).hash_stable(hcx, hasher);
                offset.hash_stable(hcx, hasher);
            }
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for GlobalAlloc<'_> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            Self::Function(f) => {
                f.hash_stable(hcx, hasher);
            }
            Self::Static(s) => {
                s.hash_stable(hcx, hasher);
            }
            Self::Memory(m) => {
                AllocationModuloRelocations(m).hash_stable(hcx, hasher);
            }
        }
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for AllocationModuloRelocations<'_> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let AllocationModuloRelocations(alloc) = *self;
        alloc.bytes.hash_stable(hcx, hasher);
        alloc.init_mask.hash_stable(hcx, hasher);
        alloc.align.hash_stable(hcx, hasher);
        alloc.mutability.hash_stable(hcx, hasher);
        alloc.extra.hash_stable(hcx, hasher);
        alloc.relocations.len().hash_stable(hcx, hasher);
        ty::tls::with_opt(|tcx| {
            if alloc.relocations.is_empty() {
                return;
            }
            let tcx = tcx.expect("can't compare AllocIds during hir lowering");
            for (size, ptr) in alloc.relocations.iter() {
                size.hash_stable(hcx, hasher);
                tcx.get_global_alloc(*ptr).hash_stable(hcx, hasher);
            }
        })
    }
}

impl PartialEq for ConstAlloc<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty
            && ty::tls::with_opt(|tcx| {
                let tcx = tcx.expect("can't compare AllocIds during hir lowering");
                tcx.get_global_alloc(self.alloc_id) == tcx.get_global_alloc(other.alloc_id)
            })
    }
}

impl PartialEq for ConstValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        ty::tls::with_opt(|tcx| match (self, other) {
            (Self::Scalar(Scalar::Int(a_int)), Self::Scalar(Scalar::Int(b_int))) => a_int == b_int,
            (
                Self::Scalar(Scalar::Ptr(a_ptr, a_size)),
                Self::Scalar(Scalar::Ptr(b_ptr, b_size)),
            ) => {
                let tcx = tcx.expect("can't hash AllocIds during hir lowering");
                let (a_ptr, a_offset) = a_ptr.into_parts();
                let (b_ptr, b_offset) = b_ptr.into_parts();
                a_offset == b_offset
                    && a_size == b_size
                    && tcx.get_global_alloc(a_ptr) == tcx.get_global_alloc(b_ptr)
            }
            (
                Self::Slice { data: a_data, start: a_start, end: a_end },
                Self::Slice { data: b_data, start: b_start, end: b_end },
            ) => {
                a_start == b_start
                    && a_end == b_end
                    && AllocationModuloRelocations(*a_data)
                        .eq(&AllocationModuloRelocations(*b_data))
            }
            (
                Self::ByRef { alloc: a_alloc, offset: a_offset },
                Self::ByRef { alloc: b_alloc, offset: b_offset },
            ) => {
                a_offset == b_offset
                    && AllocationModuloRelocations(*a_alloc)
                        .eq(&AllocationModuloRelocations(*b_alloc))
            }
            _ => false,
        })
    }
}

impl PartialEq for GlobalAlloc<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Function(a), Self::Function(b)) => a == b,
            (Self::Static(a), Self::Static(b)) => a == b,
            (Self::Memory(a), Self::Memory(b)) => {
                AllocationModuloRelocations(a).eq(&AllocationModuloRelocations(*b))
            }
            _ => false,
        }
    }
}

impl PartialEq for AllocationModuloRelocations<'_> {
    fn eq(&self, other: &Self) -> bool {
        let AllocationModuloRelocations(alloc) = *self;
        let AllocationModuloRelocations(other) = *other;
        alloc.init_mask == other.init_mask
            && alloc.align == other.align
            && alloc.mutability == other.mutability
            && alloc.extra == other.extra
            && alloc.bytes == other.bytes
            && alloc.relocations.len() == other.relocations.len()
            && ty::tls::with_opt(|tcx| {
                if alloc.relocations.is_empty() {
                    return true;
                }
                let tcx = tcx.expect("can't compare AllocIds during hir lowering");
                alloc.relocations.iter().zip(other.relocations.iter()).all(
                    |((a_size, a_ptr), (b_size, b_ptr))| {
                        a_size == b_size
                            && tcx.get_global_alloc(*a_ptr) == tcx.get_global_alloc(*b_ptr)
                    },
                )
            })
    }
}

impl Eq for ConstAlloc<'_> {}

impl Eq for ConstValue<'_> {}

impl Eq for GlobalAlloc<'_> {}

impl Eq for AllocationModuloRelocations<'_> {}

impl PartialOrd for ConstValue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for GlobalAlloc<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for AllocationModuloRelocations<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConstValue<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        ty::tls::with_opt(|tcx| {
            match (self, other) {
                (Self::Scalar(Scalar::Int(a_int)), Self::Scalar(Scalar::Int(b_int))) => {
                    a_int.cmp(&b_int)
                }
                (
                    Self::Scalar(Scalar::Ptr(a_ptr, a_size)),
                    Self::Scalar(Scalar::Ptr(b_ptr, b_size)),
                ) => {
                    let tcx = tcx.expect("can't hash AllocIds during hir lowering");
                    let (a_ptr, a_offset) = a_ptr.into_parts();
                    let (b_ptr, b_offset) = b_ptr.into_parts();
                    a_offset
                        .cmp(&b_offset)
                        .then_with(|| a_size.cmp(&b_size))
                        .then_with(|| tcx.get_global_alloc(a_ptr).cmp(&tcx.get_global_alloc(b_ptr)))
                }
                (
                    Self::Slice { data: a_data, start: a_start, end: a_end },
                    Self::Slice { data: b_data, start: b_start, end: b_end },
                ) => a_start.cmp(b_start).then_with(|| a_end.cmp(b_end)).then_with(|| {
                    AllocationModuloRelocations(*a_data).cmp(&AllocationModuloRelocations(*b_data))
                }),
                (
                    Self::ByRef { alloc: a_alloc, offset: a_offset },
                    Self::ByRef { alloc: b_alloc, offset: b_offset },
                ) => a_offset.cmp(b_offset).then_with(|| {
                    AllocationModuloRelocations(*a_alloc)
                        .cmp(&AllocationModuloRelocations(*b_alloc))
                }),
                // discriminant-based ordering
                (Self::Scalar(Scalar::Int(_)), Self::Scalar(Scalar::Ptr(..)))
                | (Self::Scalar(Scalar::Ptr(..)), Self::Scalar(Scalar::Int(_)))
                | (Self::Scalar(_), Self::Slice { .. })
                | (Self::Scalar(_), Self::ByRef { .. })
                | (Self::Slice { .. }, Self::Scalar(_)) => Ordering::Less,
                (Self::Slice { .. }, Self::ByRef { .. })
                | (Self::ByRef { .. }, Self::Scalar(_))
                | (Self::ByRef { .. }, Self::Slice { .. }) => Ordering::Greater,
            }
        })
    }
}

impl Ord for GlobalAlloc<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Function(a), Self::Function(b)) => a.cmp(b),
            (Self::Static(a), Self::Static(b)) => a.cmp(b),
            (Self::Memory(a), Self::Memory(b)) => {
                AllocationModuloRelocations(a).cmp(&AllocationModuloRelocations(b))
            }

            (Self::Function(_), Self::Static(_))
            | (Self::Function(_), Self::Memory(_))
            | (Self::Static(_), Self::Memory(_)) => Ordering::Less,

            (Self::Static(_), Self::Function(_))
            | (Self::Memory(_), Self::Function(_))
            | (Self::Memory(_), Self::Static(_)) => Ordering::Greater,
        }
    }
}

impl Ord for AllocationModuloRelocations<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let AllocationModuloRelocations(alloc) = *self;
        let AllocationModuloRelocations(other) = *other;
        alloc
            .init_mask
            .cmp(&other.init_mask)
            .then_with(|| alloc.align.cmp(&other.align))
            .then_with(|| alloc.mutability.cmp(&other.mutability))
            .then_with(|| alloc.extra.cmp(&other.extra))
            .then_with(|| alloc.bytes.cmp(&other.bytes))
            .then_with(|| alloc.relocations.len().cmp(&other.relocations.len()))
            .then_with(|| {
                ty::tls::with_opt(|tcx| {
                    if alloc.relocations.is_empty() {
                        return Ordering::Equal;
                    }
                    let tcx = tcx.expect("can't compare AllocIds during hir lowering");
                    for ((a_size, a_ptr), (b_size, b_ptr)) in
                        alloc.relocations.iter().zip(other.relocations.iter())
                    {
                        let ord = a_size.cmp(b_size).then_with(|| {
                            tcx.get_global_alloc(*a_ptr).cmp(&tcx.get_global_alloc(*b_ptr))
                        });
                        if ord != Ordering::Equal {
                            return ord;
                        }
                    }
                    Ordering::Equal
                })
            })
    }
}

impl Hash for ConstAlloc<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ty.hash(state);
        ty::tls::with_opt(|tcx| {
            let tcx = tcx.expect("can't hash AllocIds during hir lowering");
            tcx.get_global_alloc(self.alloc_id).hash(state);
        });
    }
}

impl Hash for ConstValue<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        ty::tls::with_opt(|tcx| match self {
            ConstValue::Scalar(scalar) => {
                std::mem::discriminant(scalar).hash(state);
                match scalar {
                    Scalar::Int(int) => {
                        int.hash(state);
                    }
                    Scalar::Ptr(ptr, size) => {
                        let tcx = tcx.expect("can't hash AllocIds during hir lowering");
                        let (ptr, offset) = ptr.into_parts();
                        tcx.get_global_alloc(ptr).hash(state);
                        offset.hash(state);
                        size.hash(state);
                    }
                }
            }
            ConstValue::Slice { data, start, end } => {
                AllocationModuloRelocations(*data).hash(state);
                start.hash(state);
                end.hash(state);
            }
            ConstValue::ByRef { alloc, offset } => {
                AllocationModuloRelocations(*alloc).hash(state);
                offset.hash(state);
            }
        })
    }
}

impl Hash for GlobalAlloc<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Function(f) => {
                f.hash(state);
            }
            Self::Static(s) => {
                s.hash(state);
            }
            Self::Memory(m) => {
                AllocationModuloRelocations(*m).hash(state);
            }
        }
    }
}

impl Hash for AllocationModuloRelocations<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let AllocationModuloRelocations(alloc) = *self;
        alloc.bytes.hash(state);
        alloc.init_mask.hash(state);
        alloc.align.hash(state);
        alloc.mutability.hash(state);
        alloc.extra.hash(state);
        alloc.relocations.len().hash(state);
        ty::tls::with_opt(|tcx| {
            if alloc.relocations.is_empty() {
                return;
            }
            let tcx = tcx.expect("can't compare AllocIds during hir lowering");
            for (size, ptr) in alloc.relocations.iter() {
                size.hash(state);
                tcx.get_global_alloc(*ptr).hash(state);
            }
        })
    }
}
