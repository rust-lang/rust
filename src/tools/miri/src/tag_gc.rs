use rustc_data_structures::fx::FxHashSet;

use crate::*;

pub trait VisitMachineValues {
    fn visit_machine_values(&self, visit: &mut ProvenanceVisitor);
}

pub trait MachineValue {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>);
}

pub struct ProvenanceVisitor {
    tags: FxHashSet<SbTag>,
}

impl ProvenanceVisitor {
    pub fn visit<V>(&mut self, v: V)
    where
        V: MachineValue,
    {
        v.visit_provenance(&mut self.tags);
    }
}

impl<T: MachineValue> MachineValue for &T {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        (**self).visit_provenance(tags);
    }
}

impl MachineValue for Operand<Provenance> {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        match self {
            Operand::Immediate(Immediate::Scalar(s)) => {
                s.visit_provenance(tags);
            }
            Operand::Immediate(Immediate::ScalarPair(s1, s2)) => {
                s1.visit_provenance(tags);
                s2.visit_provenance(tags);
            }
            Operand::Immediate(Immediate::Uninit) => {}
            Operand::Indirect(p) => {
                p.visit_provenance(tags);
            }
        }
    }
}

impl MachineValue for Scalar<Provenance> {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        if let Scalar::Ptr(ptr, _) = self {
            if let Provenance::Concrete { sb, .. } = ptr.provenance {
                tags.insert(sb);
            }
        }
    }
}

impl MachineValue for MemPlace<Provenance> {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        if let Some(Provenance::Concrete { sb, .. }) = self.ptr.provenance {
            tags.insert(sb);
        }
    }
}

impl MachineValue for SbTag {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        tags.insert(*self);
    }
}

impl MachineValue for Pointer<Provenance> {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        let (prov, _offset) = self.into_parts();
        if let Provenance::Concrete { sb, .. } = prov {
            tags.insert(sb);
        }
    }
}

impl MachineValue for Pointer<Option<Provenance>> {
    fn visit_provenance(&self, tags: &mut FxHashSet<SbTag>) {
        let (prov, _offset) = self.into_parts();
        if let Some(Provenance::Concrete { sb, .. }) = prov {
            tags.insert(sb);
        }
    }
}

impl VisitMachineValues for Allocation<Provenance, AllocExtra> {
    fn visit_machine_values(&self, visit: &mut ProvenanceVisitor) {
        for (_size, prov) in self.provenance().iter() {
            if let Provenance::Concrete { sb, .. } = prov {
                visit.visit(*sb);
            }
        }

        self.extra.visit_machine_values(visit);
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: MiriInterpCxExt<'mir, 'tcx> {
    /// GC helper to visit everything that can store provenance. The `ProvenanceVisitor` knows how
    /// to extract provenance from the interpreter data types.
    fn visit_all_machine_values(&self, acc: &mut ProvenanceVisitor) {
        let this = self.eval_context_ref();

        // Memory.
        this.memory.alloc_map().iter(|it| {
            for (_id, (_kind, alloc)) in it {
                alloc.visit_machine_values(acc);
            }
        });

        // And all the other machine values.
        this.machine.visit_machine_values(acc);
    }

    fn garbage_collect_tags(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // No reason to do anything at all if stacked borrows is off.
        if this.machine.stacked_borrows.is_none() {
            return Ok(());
        }

        let mut visitor = ProvenanceVisitor { tags: FxHashSet::default() };
        this.visit_all_machine_values(&mut visitor);
        self.remove_unreachable_tags(visitor.tags);

        Ok(())
    }

    fn remove_unreachable_tags(&mut self, tags: FxHashSet<SbTag>) {
        let this = self.eval_context_mut();
        this.memory.alloc_map().iter(|it| {
            for (_id, (_kind, alloc)) in it {
                alloc
                    .extra
                    .stacked_borrows
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .remove_unreachable_tags(&tags);
            }
        });
    }
}
