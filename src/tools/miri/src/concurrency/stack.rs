use crate::*;

/// An encapsulated call stack, so encapsulated to enable efficient caching of metadata about the
/// contained `Frame`.
pub struct Stack<'mir, 'tcx> {
    frames: Vec<Frame<'mir, 'tcx, Provenance, FrameData<'tcx>>>,
}

impl<'mir, 'tcx> Stack<'mir, 'tcx> {
    pub fn new() -> Self {
        Stack { frames: Vec::new() }
    }

    /// Does this `Stack` contain any `Frame`s?
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Borrow the call frames of a `Stack`.
    pub fn frames(&self) -> &[Frame<'mir, 'tcx, Provenance, FrameData<'tcx>>] {
        &self.frames[..]
    }

    /// Mutably borrow the call frames of a `Stack`.
    pub fn frames_mut(&mut self) -> &mut [Frame<'mir, 'tcx, Provenance, FrameData<'tcx>>] {
        &mut self.frames[..]
    }

    /// Push a new `Frame` onto a `Stack`.
    pub fn push(&mut self, frame: Frame<'mir, 'tcx, Provenance, FrameData<'tcx>>) {
        self.frames.push(frame);
    }

    /// Try to pop a `Frame` from a `Stack`.
    pub fn pop(&mut self) -> Option<Frame<'mir, 'tcx, Provenance, FrameData<'tcx>>> {
        self.frames.pop()
    }
}

impl VisitTags for Stack<'_, '_> {
    fn visit_tags(&self, visit: &mut dyn FnMut(SbTag)) {
        let Stack { frames } = self;
        for frame in frames {
            frame.visit_tags(visit);
        }
    }
}
