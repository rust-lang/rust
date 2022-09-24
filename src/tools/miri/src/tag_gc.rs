use rustc_data_structures::fx::FxHashSet;

use crate::*;

pub trait VisitMachineValues {
    fn visit_machine_values(&self, visit: &mut impl FnMut(&Operand<Provenance>));
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: MiriInterpCxExt<'mir, 'tcx> {
    /// Generic GC helper to visit everything that can store a value. The `acc` offers some chance to
    /// accumulate everything.
    fn visit_all_machine_values<T>(
        &self,
        acc: &mut T,
        mut visit_operand: impl FnMut(&mut T, &Operand<Provenance>),
        mut visit_alloc: impl FnMut(&mut T, &Allocation<Provenance, AllocExtra>),
    ) {
        let this = self.eval_context_ref();

        // Memory.
        this.memory.alloc_map().iter(|it| {
            for (_id, (_kind, alloc)) in it {
                visit_alloc(acc, alloc);
            }
        });

        // And all the other machine values.
        this.machine.visit_machine_values(&mut |op| visit_operand(acc, op));
    }

    fn garbage_collect_tags(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // No reason to do anything at all if stacked borrows is off.
        if this.machine.stacked_borrows.is_none() {
            return Ok(());
        }

        let mut tags = FxHashSet::default();

        let visit_scalar = |tags: &mut FxHashSet<SbTag>, s: &Scalar<Provenance>| {
            if let Scalar::Ptr(ptr, _) = s {
                if let Provenance::Concrete { sb, .. } = ptr.provenance {
                    tags.insert(sb);
                }
            }
        };

        this.visit_all_machine_values(
            &mut tags,
            |tags, op| {
                match op {
                    Operand::Immediate(Immediate::Scalar(s)) => {
                        visit_scalar(tags, s);
                    }
                    Operand::Immediate(Immediate::ScalarPair(s1, s2)) => {
                        visit_scalar(tags, s1);
                        visit_scalar(tags, s2);
                    }
                    Operand::Immediate(Immediate::Uninit) => {}
                    Operand::Indirect(MemPlace { ptr, .. }) => {
                        if let Some(Provenance::Concrete { sb, .. }) = ptr.provenance {
                            tags.insert(sb);
                        }
                    }
                }
            },
            |tags, alloc| {
                for (_size, prov) in alloc.provenance().iter() {
                    if let Provenance::Concrete { sb, .. } = prov {
                        tags.insert(*sb);
                    }
                }
                let stacks = alloc
                    .extra
                    .stacked_borrows
                    .as_ref()
                    .expect("we should not even enter the GC if Stacked Borrows is disabled");
                tags.extend(&stacks.borrow().exposed_tags);

                if let Some(store_buffers) = alloc.extra.weak_memory.as_ref() {
                    store_buffers.iter(|val| {
                        if let Scalar::Ptr(ptr, _) = val {
                            if let Provenance::Concrete { sb, .. } = ptr.provenance {
                                tags.insert(sb);
                            }
                        }
                    });
                }
            },
        );

        self.remove_unreachable_tags(tags);

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
