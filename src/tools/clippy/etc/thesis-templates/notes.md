```rs
fn visit_terminator(&mut self, term: &mir::Terminator<'tcx>, loc: mir::Location) {
    match &term.kind {
        mir::TerminatorKind::Drop { place, .. } => {
            if place.local == self.local && self.states[loc.block].valid() {
                self.states[loc.block] = State::Dropped;
            }
        },
        mir::TerminatorKind::SwitchInt { discr: op, .. } | mir::TerminatorKind::Assert { cond: op, .. } => {
            if let Some(place) = op.place()
                && place.local == self.local
            {
                todo!();
            }
        },
        mir::TerminatorKind::Call {
            func,
            args,
            destination: dest,
            ..
        } => {
            if let Some(place) = func.place()
                && place.local == self.local
            {
                todo!();
            }

            for arg in args {
                if let Some(place) = arg.node.place()
                    && place.local == self.local
                {
                    todo!();
                }
            }

            if dest.local == self.local {
                todo!()
            }
        },

        // Controll flow or unstable features. Uninteresting for values
        mir::TerminatorKind::Goto { .. }
        | mir::TerminatorKind::UnwindResume
        | mir::TerminatorKind::UnwindTerminate(_)
        | mir::TerminatorKind::Return
        | mir::TerminatorKind::Unreachable
        | mir::TerminatorKind::Yield { .. }
        | mir::TerminatorKind::CoroutineDrop
        | mir::TerminatorKind::FalseEdge { .. }
        | mir::TerminatorKind::FalseUnwind { .. }
        | mir::TerminatorKind::InlineAsm { .. } => {},
    }
    self.super_terminator(term, loc)
}
``` 
