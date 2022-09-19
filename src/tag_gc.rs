use crate::*;
use rustc_data_structures::fx::FxHashSet;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: MiriInterpCxExt<'mir, 'tcx> {
    fn garbage_collect_tags(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        // No reason to do anything at all if stacked borrows is off.
        if this.machine.stacked_borrows.is_none() {
            return Ok(());
        }

        let mut tags = FxHashSet::default();

        for thread in this.machine.threads.iter() {
            if let Some(Scalar::Ptr(
                Pointer { provenance: Provenance::Concrete { sb, .. }, .. },
                _,
            )) = thread.panic_payload
            {
                tags.insert(sb);
            }
        }

        self.find_tags_in_tls(&mut tags);
        self.find_tags_in_memory(&mut tags);
        self.find_tags_in_locals(&mut tags)?;

        self.remove_unreachable_tags(tags);

        Ok(())
    }

    fn find_tags_in_tls(&mut self, tags: &mut FxHashSet<SbTag>) {
        let this = self.eval_context_mut();
        this.machine.tls.iter(|scalar| {
            if let Scalar::Ptr(Pointer { provenance: Provenance::Concrete { sb, .. }, .. }, _) =
                scalar
            {
                tags.insert(*sb);
            }
        });
    }

    fn find_tags_in_memory(&mut self, tags: &mut FxHashSet<SbTag>) {
        let this = self.eval_context_mut();
        this.memory.alloc_map().iter(|it| {
            for (_id, (_kind, alloc)) in it {
                for (_size, prov) in alloc.provenance().iter() {
                    if let Provenance::Concrete { sb, .. } = prov {
                        tags.insert(*sb);
                    }
                }
            }
        });
    }

    fn find_tags_in_locals(&mut self, tags: &mut FxHashSet<SbTag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        for frame in this.machine.threads.all_stacks().flatten() {
            // Handle the return place of each frame
            if let Ok(return_place) = frame.return_place.try_as_mplace() {
                if let Some(Provenance::Concrete { sb, .. }) = return_place.ptr.provenance {
                    tags.insert(sb);
                }
            }

            for local in frame.locals.iter() {
                let LocalValue::Live(value) = local.value else {
                continue;
            };
                match value {
                    Operand::Immediate(Immediate::Scalar(Scalar::Ptr(ptr, _))) =>
                        if let Provenance::Concrete { sb, .. } = ptr.provenance {
                            tags.insert(sb);
                        },
                    Operand::Immediate(Immediate::ScalarPair(s1, s2)) => {
                        if let Scalar::Ptr(ptr, _) = s1 {
                            if let Provenance::Concrete { sb, .. } = ptr.provenance {
                                tags.insert(sb);
                            }
                        }
                        if let Scalar::Ptr(ptr, _) = s2 {
                            if let Provenance::Concrete { sb, .. } = ptr.provenance {
                                tags.insert(sb);
                            }
                        }
                    }
                    Operand::Indirect(MemPlace { ptr, .. }) => {
                        if let Some(Provenance::Concrete { sb, .. }) = ptr.provenance {
                            tags.insert(sb);
                        }
                    }
                    Operand::Immediate(Immediate::Uninit)
                    | Operand::Immediate(Immediate::Scalar(Scalar::Int(_))) => {}
                }
            }
        }

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
