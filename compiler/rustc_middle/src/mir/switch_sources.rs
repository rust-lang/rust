//! Lazily compute the inverse of each `SwitchInt`'s switch targets. Modeled after
//! `Predecessors`/`PredecessorCache`.

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::stable_map::FxHashMap;
use rustc_data_structures::sync::OnceCell;
use rustc_index::vec::IndexVec;
use rustc_serialize as serialize;
use smallvec::SmallVec;

use crate::mir::{BasicBlock, BasicBlockData, Terminator, TerminatorKind};

pub type SwitchSources = FxHashMap<(BasicBlock, BasicBlock), SmallVec<[Option<u128>; 1]>>;

#[derive(Clone, Debug)]
pub(super) struct SwitchSourceCache {
    cache: OnceCell<SwitchSources>,
}

impl SwitchSourceCache {
    #[inline]
    pub(super) fn new() -> Self {
        SwitchSourceCache { cache: OnceCell::new() }
    }

    /// Invalidates the switch source cache.
    #[inline]
    pub(super) fn invalidate(&mut self) {
        self.cache = OnceCell::new();
    }

    /// Returns the switch sources for this MIR.
    #[inline]
    pub(super) fn compute(
        &self,
        basic_blocks: &IndexVec<BasicBlock, BasicBlockData<'_>>,
    ) -> &SwitchSources {
        self.cache.get_or_init(|| {
            let mut switch_sources: SwitchSources = FxHashMap::default();
            for (bb, data) in basic_blocks.iter_enumerated() {
                if let Some(Terminator {
                    kind: TerminatorKind::SwitchInt { targets, .. }, ..
                }) = &data.terminator
                {
                    for (value, target) in targets.iter() {
                        switch_sources.entry((target, bb)).or_default().push(Some(value));
                    }
                    switch_sources.entry((targets.otherwise(), bb)).or_default().push(None);
                }
            }

            switch_sources
        })
    }
}

impl<S: serialize::Encoder> serialize::Encodable<S> for SwitchSourceCache {
    #[inline]
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_unit()
    }
}

impl<D: serialize::Decoder> serialize::Decodable<D> for SwitchSourceCache {
    #[inline]
    fn decode(_: &mut D) -> Self {
        Self::new()
    }
}

impl<CTX> HashStable<CTX> for SwitchSourceCache {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, _: &mut StableHasher) {
        // do nothing
    }
}

TrivialTypeFoldableAndLiftImpls! {
    SwitchSourceCache,
}
