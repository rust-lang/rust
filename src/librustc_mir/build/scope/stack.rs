use crate::build::scope::{BreakableScope, BreakableTarget, DropKind};
use rustc::middle::region;
use rustc::mir::{BasicBlock, Local, Place, SourceScope, SourceInfo};
use syntax_pos::Span;
use rustc_data_structures::fx::FxHashMap;

#[derive(Debug)]
crate struct Scope {
    /// The source scope this scope was created in.
    pub(super) source_scope: SourceScope,

    /// the region span of this scope within source code.
    pub(super) region_scope: region::Scope,

    /// the span of that region_scope
    pub(super) region_scope_span: Span,

    /// Whether there's anything to do for the cleanup path, that is,
    /// when unwinding through this scope. This includes destructors,
    /// but not StorageDead statements, which don't get emitted at all
    /// for unwinding, for several reasons:
    ///  * clang doesn't emit llvm.lifetime.end for C++ unwinding
    ///  * LLVM's memory dependency analysis can't handle it atm
    ///  * polluting the cleanup MIR with StorageDead creates
    ///    landing pads even though there's no actual destructors
    ///  * freeing up stack space has no effect during unwinding
    /// Note that for generators we do emit StorageDeads, for the
    /// use of optimizations in the MIR generator transform.
    pub(super) needs_cleanup: bool,

    /// set of places to drop when exiting this scope. This starts
    /// out empty but grows as variables are declared during the
    /// building process. This is a stack, so we always drop from the
    /// end of the vector (top of the stack) first.
    pub(super) drops: Vec<DropData>,

    /// The cache for drop chain on “normal” exit into a particular BasicBlock.
    pub(super) cached_exits: FxHashMap<(BasicBlock, region::Scope), BasicBlock>,

    /// The cache for drop chain on "generator drop" exit.
    pub(super) cached_generator_drop: Option<BasicBlock>,

    /// The cache for drop chain on "unwind" exit.
    pub(super) cached_unwind: CachedBlock,
}

impl Scope {
    /// Invalidates all the cached blocks in the scope.
    ///
    /// Should always be run for all inner scopes when a drop is pushed into some scope enclosing a
    /// larger extent of code.
    ///
    /// `storage_only` controls whether to invalidate only drop paths that run `StorageDead`.
    /// `this_scope_only` controls whether to invalidate only drop paths that refer to the current
    /// top-of-scope (as opposed to dependent scopes).
    pub(super) fn invalidate_cache(&mut self, storage_only: bool, is_generator: bool, this_scope_only: bool) {
        // FIXME: maybe do shared caching of `cached_exits` etc. to handle functions
        // with lots of `try!`?

        // cached exits drop storage and refer to the top-of-scope
        self.cached_exits.clear();

        // the current generator drop and unwind refer to top-of-scope
        self.cached_generator_drop = None;

        let ignore_unwinds = storage_only && !is_generator;
        if !ignore_unwinds {
            self.cached_unwind.invalidate();
        }

        if !ignore_unwinds && !this_scope_only {
            for drop_data in &mut self.drops {
                drop_data.cached_block.invalidate();
            }
        }
    }

    /// Given a span and this scope's source scope, make a SourceInfo.
    pub(super) fn source_info(&self, span: Span) -> SourceInfo {
        SourceInfo {
            span,
            scope: self.source_scope
        }
    }
}

#[derive(Debug)]
pub(super) struct DropData {
    /// span where drop obligation was incurred (typically where place was declared)
    pub(super) span: Span,

    /// local to drop
    pub(super) local: Local,

    /// Whether this is a value Drop or a StorageDead.
    pub(super) kind: DropKind,

    /// The cached blocks for unwinds.
    pub(super) cached_block: CachedBlock,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CachedBlock {
    /// The cached block for the cleanups-on-diverge path. This block
    /// contains code to run the current drop and all the preceding
    /// drops (i.e., those having lower index in Drop’s Scope drop
    /// array)
    unwind: Option<BasicBlock>,

    /// The cached block for unwinds during cleanups-on-generator-drop path
    ///
    /// This is split from the standard unwind path here to prevent drop
    /// elaboration from creating drop flags that would have to be captured
    /// by the generator. I'm not sure how important this optimization is,
    /// but it is here.
    generator_drop: Option<BasicBlock>,
}

impl CachedBlock {
    fn invalidate(&mut self) {
        self.generator_drop = None;
        self.unwind = None;
    }

    pub(super) fn get(&self, generator_drop: bool) -> Option<BasicBlock> {
        if generator_drop {
            self.generator_drop
        } else {
            self.unwind
        }
    }

    pub(super) fn ref_mut(&mut self, generator_drop: bool) -> &mut Option<BasicBlock> {
        if generator_drop {
            &mut self.generator_drop
        } else {
            &mut self.unwind
        }
    }
}

#[derive(Debug, Default)]
pub struct Scopes<'tcx> {
    pub(super) scopes: Vec<Scope>,
    /// The current set of breakable scopes. See module comment for more details.
    pub(super) breakable_scopes: Vec<BreakableScope<'tcx>>,
}

impl<'tcx> Scopes<'tcx> {
    pub(super) fn len(&self) -> usize {
        self.scopes.len()
    }

    pub(super) fn push_scope(&mut self, region_scope: (region::Scope, SourceInfo), vis_scope: SourceScope) {
        debug!("push_scope({:?})", region_scope);
        self.scopes.push(Scope {
            source_scope: vis_scope,
            region_scope: region_scope.0,
            region_scope_span: region_scope.1.span,
            needs_cleanup: false,
            drops: vec![],
            cached_generator_drop: None,
            cached_exits: Default::default(),
            cached_unwind: CachedBlock::default(),
        });
    }

    pub(super) fn pop_scope(
        &mut self,
        region_scope: (region::Scope, SourceInfo),
    ) -> (Scope, Option<BasicBlock>) {
        let scope = self.scopes.pop().unwrap();
        assert_eq!(scope.region_scope, region_scope.0);
        let unwind_to = self.scopes.last()
            .and_then(|next_scope| next_scope.cached_unwind.get(false));
        (scope, unwind_to)
    }

    pub(super) fn may_panic(&self, scope_count: usize) -> bool {
        let len = self.len();
        self.scopes[(len - scope_count)..].iter().any(|s| s.needs_cleanup)
    }

    /// Finds the breakable scope for a given label. This is used for
    /// resolving `return`, `break` and `continue`.
    pub(super) fn find_breakable_scope(
        &self,
        span: Span,
        target: BreakableTarget,
    ) -> (BasicBlock, region::Scope, Option<Place<'tcx>>) {
        let get_scope = |scope: region::Scope| {
            // find the loop-scope by its `region::Scope`.
            self.breakable_scopes.iter()
                .rfind(|breakable_scope| breakable_scope.region_scope == scope)
                .unwrap_or_else(|| span_bug!(span, "no enclosing breakable scope found"))
        };
        match target {
            BreakableTarget::Return => {
                let scope = &self.breakable_scopes[0];
                if scope.break_destination != Place::return_place() {
                    span_bug!(span, "`return` in item with no return scope");
                }
                (scope.break_block, scope.region_scope, Some(scope.break_destination.clone()))
            }
            BreakableTarget::Break(scope) => {
                let scope = get_scope(scope);
                (scope.break_block, scope.region_scope, Some(scope.break_destination.clone()))
            }
            BreakableTarget::Continue(scope) => {
                let scope = get_scope(scope);
                let continue_block = scope.continue_block
                    .unwrap_or_else(|| span_bug!(span, "missing `continue` block"));
                (continue_block, scope.region_scope, None)
            }
        }
    }

    pub(super) fn num_scopes_above(&self, region_scope: region::Scope, span: Span) -> usize {
        let scope_count = self.scopes.iter().rev()
            .position(|scope| scope.region_scope == region_scope)
            .unwrap_or_else(|| {
                span_bug!(span, "region_scope {:?} does not enclose", region_scope)
            });
        let len = self.len();
        assert!(scope_count < len, "should not use `exit_scope` to pop ALL scopes");
        scope_count
    }

    pub(super) fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item=&mut Scope> + '_ {
        self.scopes.iter_mut().rev()
    }

    pub(super) fn top_scopes(&mut self, count: usize) -> impl DoubleEndedIterator<Item=&mut Scope> + '_ {
        let len = self.len();
        self.scopes[len - count..].iter_mut()
    }

    /// Returns the topmost active scope, which is known to be alive until
    /// the next scope expression.
    crate fn topmost(&self) -> region::Scope {
        self.scopes.last().expect("topmost_scope: no scopes present").region_scope
    }

    pub(super) fn source_info(&self, index: usize, span: Span) -> SourceInfo {
        self.scopes[self.len() - index].source_info(span)
    }
}
