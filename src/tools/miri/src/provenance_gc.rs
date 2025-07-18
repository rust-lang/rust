use either::Either;
use rustc_data_structures::fx::FxHashSet;

use crate::*;

pub type VisitWith<'a> = dyn FnMut(Option<AllocId>, Option<BorTag>) + 'a;

pub trait VisitProvenance {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>);
}

// Trivial impls for types that do not contain any provenance
macro_rules! no_provenance {
    ($($ty:ident)+) => {
        $(
            impl VisitProvenance for $ty {
                fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
            }
        )+
    }
}
no_provenance!(i8 i16 i32 i64 isize u8 u16 u32 u64 usize ThreadId);

impl<T: VisitProvenance> VisitProvenance for Option<T> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        if let Some(x) = self {
            x.visit_provenance(visit);
        }
    }
}

impl<A, B> VisitProvenance for (A, B)
where
    A: VisitProvenance,
    B: VisitProvenance,
{
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        self.0.visit_provenance(visit);
        self.1.visit_provenance(visit);
    }
}

impl<T: VisitProvenance> VisitProvenance for std::cell::RefCell<T> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        self.borrow().visit_provenance(visit)
    }
}

impl VisitProvenance for BorTag {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        visit(None, Some(*self))
    }
}

impl VisitProvenance for AllocId {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        visit(Some(*self), None)
    }
}

impl VisitProvenance for Provenance {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        if let Provenance::Concrete { alloc_id, tag, .. } = self {
            visit(Some(*alloc_id), Some(*tag));
        }
    }
}

impl VisitProvenance for StrictPointer {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        self.provenance.visit_provenance(visit);
    }
}

impl VisitProvenance for Pointer {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        self.provenance.visit_provenance(visit);
    }
}

impl VisitProvenance for Scalar {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self {
            Scalar::Ptr(ptr, _) => ptr.visit_provenance(visit),
            Scalar::Int(_) => (),
        }
    }
}

impl VisitProvenance for IoError {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        use crate::shims::io_error::IoError::*;
        match self {
            LibcError(_name) => (),
            WindowsError(_name) => (),
            HostError(_io_error) => (),
            Raw(scalar) => scalar.visit_provenance(visit),
        }
    }
}

impl VisitProvenance for Immediate<Provenance> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self {
            Immediate::Scalar(s) => {
                s.visit_provenance(visit);
            }
            Immediate::ScalarPair(s1, s2) => {
                s1.visit_provenance(visit);
                s2.visit_provenance(visit);
            }
            Immediate::Uninit => {}
        }
    }
}

impl VisitProvenance for MemPlaceMeta<Provenance> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self {
            MemPlaceMeta::Meta(m) => m.visit_provenance(visit),
            MemPlaceMeta::None => {}
        }
    }
}

impl VisitProvenance for ImmTy<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        (**self).visit_provenance(visit)
    }
}

impl VisitProvenance for MPlaceTy<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        self.ptr().visit_provenance(visit);
        self.meta().visit_provenance(visit);
    }
}

impl VisitProvenance for PlaceTy<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self.as_mplace_or_local() {
            Either::Left(mplace) => mplace.visit_provenance(visit),
            Either::Right(_) => (),
        }
    }
}

impl VisitProvenance for OpTy<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self.as_mplace_or_imm() {
            Either::Left(mplace) => mplace.visit_provenance(visit),
            Either::Right(imm) => imm.visit_provenance(visit),
        }
    }
}

impl VisitProvenance for Allocation<Provenance, AllocExtra<'_>, MiriAllocBytes> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        for prov in self.provenance().provenances() {
            prov.visit_provenance(visit);
        }

        self.extra.visit_provenance(visit);
    }
}

impl VisitProvenance for crate::MiriInterpCx<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        // Visit the contents of the allocations and the IDs themselves, to account for all
        // live allocation IDs and all provenance in the allocation bytes, even if they are leaked.
        // We do *not* visit all the `AllocId` of the live allocations; we tried that and adding
        // them all to the live set is too expensive. Instead we later do liveness check by
        // checking both "is this alloc id live" and "is it mentioned anywhere else in
        // the interpreter state".
        self.memory.alloc_map().iter(|it| {
            for (_id, (_kind, alloc)) in it {
                alloc.visit_provenance(visit);
            }
        });
        // And all the other machine values.
        self.machine.visit_provenance(visit);
    }
}

pub struct LiveAllocs<'a, 'tcx> {
    collected: FxHashSet<AllocId>,
    ecx: &'a MiriInterpCx<'tcx>,
}

impl LiveAllocs<'_, '_> {
    pub fn is_live(&self, id: AllocId) -> bool {
        self.collected.contains(&id) || self.ecx.is_alloc_live(id)
    }
}

fn remove_unreachable_tags<'tcx>(ecx: &mut MiriInterpCx<'tcx>, tags: FxHashSet<BorTag>) {
    // Avoid iterating all allocations if there's no borrow tracker anyway.
    if ecx.machine.borrow_tracker.is_some() {
        ecx.memory.alloc_map().iter(|it| {
            for (_id, (_kind, alloc)) in it {
                alloc.extra.borrow_tracker.as_ref().unwrap().remove_unreachable_tags(&tags);
            }
        });
    }
}

fn remove_unreachable_allocs<'tcx>(ecx: &mut MiriInterpCx<'tcx>, allocs: FxHashSet<AllocId>) {
    let allocs = LiveAllocs { ecx, collected: allocs };
    ecx.machine.allocation_spans.borrow_mut().retain(|id, _| allocs.is_live(*id));
    ecx.machine.symbolic_alignment.borrow_mut().retain(|id, _| allocs.is_live(*id));
    ecx.machine.alloc_addresses.borrow_mut().remove_unreachable_allocs(&allocs);
    if let Some(borrow_tracker) = &ecx.machine.borrow_tracker {
        borrow_tracker.borrow_mut().remove_unreachable_allocs(&allocs);
    }
    // Clean up core (non-Miri-specific) state.
    ecx.remove_unreachable_allocs(&allocs.collected);
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: MiriInterpCxExt<'tcx> {
    fn run_provenance_gc(&mut self) {
        let this = self.eval_context_mut();

        // We collect all tags and AllocId from every part of the interpreter.
        let mut tags = FxHashSet::default();
        let mut alloc_ids = FxHashSet::default();
        this.visit_provenance(&mut |id, tag| {
            if let Some(id) = id {
                alloc_ids.insert(id);
            }
            if let Some(tag) = tag {
                tags.insert(tag);
            }
        });

        // Based on this, clean up the interpreter state.
        remove_unreachable_tags(this, tags);
        remove_unreachable_allocs(this, alloc_ids);
    }
}
