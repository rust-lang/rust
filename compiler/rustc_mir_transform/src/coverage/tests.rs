//! This crate hosts a selection of "unit tests" for components of the `InstrumentCoverage` MIR
//! pass.
//!
//! ```shell
//! ./x.py test --keep-stage 1 compiler/rustc_mir --test-args '--show-output coverage'
//! ```
//!
//! The tests construct a few "mock" objects, as needed, to support the `InstrumentCoverage`
//! functions and algorithms. Mocked objects include instances of `mir::Body`; including
//! `Terminator`s of various `kind`s, and `Span` objects. Some functions used by or used on
//! real, runtime versions of these mocked-up objects have constraints (such as cross-thread
//! limitations) and deep dependencies on other elements of the full Rust compiler (which is
//! *not* constructed or mocked for these tests).
//!
//! Of particular note, attempting to simply print elements of the `mir::Body` with default
//! `Debug` formatting can fail because some `Debug` format implementations require the
//! `TyCtxt`, obtained via a static global variable that is *not* set for these tests.
//! Initializing the global type context is prohibitively complex for the scope and scale of these
//! tests (essentially requiring initializing the entire compiler).
//!
//! Also note, some basic features of `Span` also rely on the `Span`s own "session globals", which
//! are unrelated to the `TyCtxt` global. Without initializing the `Span` session globals, some
//! basic, coverage-specific features would be impossible to test, but thankfully initializing these
//! globals is comparatively simpler. The easiest way is to wrap the test in a closure argument
//! to: `rustc_span::create_default_session_globals_then(|| { test_here(); })`.

use itertools::Itertools;
use rustc_data_structures::graph::{DirectedGraph, Successors};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::{bug, ty};
use rustc_span::{BytePos, DUMMY_SP, Pos, Span};

use super::graph::{self, BasicCoverageBlock};

fn bcb(index: u32) -> BasicCoverageBlock {
    BasicCoverageBlock::from_u32(index)
}

// All `TEMP_BLOCK` targets should be replaced before calling `to_body() -> mir::Body`.
const TEMP_BLOCK: BasicBlock = BasicBlock::MAX;

struct MockBlocks<'tcx> {
    blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    dummy_place: Place<'tcx>,
    next_local: usize,
}

impl<'tcx> MockBlocks<'tcx> {
    fn new() -> Self {
        Self {
            blocks: IndexVec::new(),
            dummy_place: Place { local: RETURN_PLACE, projection: ty::List::empty() },
            next_local: 0,
        }
    }

    fn new_temp(&mut self) -> Local {
        let index = self.next_local;
        self.next_local += 1;
        Local::new(index)
    }

    fn push(&mut self, kind: TerminatorKind<'tcx>) -> BasicBlock {
        let next_lo = if let Some(last) = self.blocks.last_index() {
            self.blocks[last].terminator().source_info.span.hi()
        } else {
            BytePos(1)
        };
        let next_hi = next_lo + BytePos(1);
        self.blocks.push(BasicBlockData::new(
            Some(Terminator {
                source_info: SourceInfo::outermost(Span::with_root_ctxt(next_lo, next_hi)),
                kind,
            }),
            false,
        ))
    }

    fn link(&mut self, from_block: BasicBlock, to_block: BasicBlock) {
        match self.blocks[from_block].terminator_mut().kind {
            TerminatorKind::Assert { ref mut target, .. }
            | TerminatorKind::Call { target: Some(ref mut target), .. }
            | TerminatorKind::Drop { ref mut target, .. }
            | TerminatorKind::FalseEdge { real_target: ref mut target, .. }
            | TerminatorKind::FalseUnwind { real_target: ref mut target, .. }
            | TerminatorKind::Goto { ref mut target }
            | TerminatorKind::Yield { resume: ref mut target, .. } => *target = to_block,
            ref invalid => bug!("Invalid from_block: {:?}", invalid),
        }
    }

    fn add_block_from(
        &mut self,
        some_from_block: Option<BasicBlock>,
        to_kind: TerminatorKind<'tcx>,
    ) -> BasicBlock {
        let new_block = self.push(to_kind);
        if let Some(from_block) = some_from_block {
            self.link(from_block, new_block);
        }
        new_block
    }

    fn set_branch(&mut self, switchint: BasicBlock, branch_index: usize, to_block: BasicBlock) {
        match self.blocks[switchint].terminator_mut().kind {
            TerminatorKind::SwitchInt { ref mut targets, .. } => {
                let mut branches = targets.iter().collect::<Vec<_>>();
                let otherwise = if branch_index == branches.len() {
                    to_block
                } else {
                    let old_otherwise = targets.otherwise();
                    if branch_index > branches.len() {
                        branches.push((branches.len() as u128, old_otherwise));
                        while branches.len() < branch_index {
                            branches.push((branches.len() as u128, TEMP_BLOCK));
                        }
                        to_block
                    } else {
                        branches[branch_index] = (branch_index as u128, to_block);
                        old_otherwise
                    }
                };
                *targets = SwitchTargets::new(branches.into_iter(), otherwise);
            }
            ref invalid => bug!("Invalid BasicBlock kind or no to_block: {:?}", invalid),
        }
    }

    fn call(&mut self, some_from_block: Option<BasicBlock>) -> BasicBlock {
        self.add_block_from(
            some_from_block,
            TerminatorKind::Call {
                func: Operand::Copy(self.dummy_place.clone()),
                args: [].into(),
                destination: self.dummy_place.clone(),
                target: Some(TEMP_BLOCK),
                unwind: UnwindAction::Continue,
                call_source: CallSource::Misc,
                fn_span: DUMMY_SP,
            },
        )
    }

    fn goto(&mut self, some_from_block: Option<BasicBlock>) -> BasicBlock {
        self.add_block_from(some_from_block, TerminatorKind::Goto { target: TEMP_BLOCK })
    }

    fn switchint(&mut self, some_from_block: Option<BasicBlock>) -> BasicBlock {
        let switchint_kind = TerminatorKind::SwitchInt {
            discr: Operand::Move(Place::from(self.new_temp())),
            targets: SwitchTargets::static_if(0, TEMP_BLOCK, TEMP_BLOCK),
        };
        self.add_block_from(some_from_block, switchint_kind)
    }

    fn return_(&mut self, some_from_block: Option<BasicBlock>) -> BasicBlock {
        self.add_block_from(some_from_block, TerminatorKind::Return)
    }

    fn to_body(self) -> Body<'tcx> {
        Body::new_cfg_only(self.blocks)
    }
}

fn debug_basic_blocks(mir_body: &Body<'_>) -> String {
    format!(
        "{:?}",
        mir_body
            .basic_blocks
            .iter_enumerated()
            .map(|(bb, data)| {
                let term = &data.terminator();
                let kind = &term.kind;
                let span = term.source_info.span;
                let sp = format!("(span:{},{})", span.lo().to_u32(), span.hi().to_u32());
                match kind {
                    TerminatorKind::Assert { target, .. }
                    | TerminatorKind::Call { target: Some(target), .. }
                    | TerminatorKind::Drop { target, .. }
                    | TerminatorKind::FalseEdge { real_target: target, .. }
                    | TerminatorKind::FalseUnwind { real_target: target, .. }
                    | TerminatorKind::Goto { target }
                    | TerminatorKind::Yield { resume: target, .. } => {
                        format!("{}{:?}:{} -> {:?}", sp, bb, kind.name(), target)
                    }
                    TerminatorKind::InlineAsm { targets, .. } => {
                        format!("{}{:?}:{} -> {:?}", sp, bb, kind.name(), targets)
                    }
                    TerminatorKind::SwitchInt { targets, .. } => {
                        format!("{}{:?}:{} -> {:?}", sp, bb, kind.name(), targets)
                    }
                    _ => format!("{}{:?}:{}", sp, bb, kind.name()),
                }
            })
            .collect::<Vec<_>>()
    )
}

static PRINT_GRAPHS: bool = false;

fn print_mir_graphviz(name: &str, mir_body: &Body<'_>) {
    if PRINT_GRAPHS {
        println!(
            "digraph {} {{\n{}\n}}",
            name,
            mir_body
                .basic_blocks
                .iter_enumerated()
                .map(|(bb, data)| {
                    format!(
                        "    {:?} [label=\"{:?}: {}\"];\n{}",
                        bb,
                        bb,
                        data.terminator().kind.name(),
                        mir_body
                            .basic_blocks
                            .successors(bb)
                            .map(|successor| { format!("    {:?} -> {:?};", bb, successor) })
                            .join("\n")
                    )
                })
                .join("\n")
        );
    }
}

fn print_coverage_graphviz(name: &str, mir_body: &Body<'_>, graph: &graph::CoverageGraph) {
    if PRINT_GRAPHS {
        println!(
            "digraph {} {{\n{}\n}}",
            name,
            graph
                .iter_enumerated()
                .map(|(bcb, bcb_data)| {
                    format!(
                        "    {:?} [label=\"{:?}: {}\"];\n{}",
                        bcb,
                        bcb,
                        mir_body[bcb_data.last_bb()].terminator().kind.name(),
                        graph
                            .successors(bcb)
                            .map(|successor| { format!("    {:?} -> {:?};", bcb, successor) })
                            .join("\n")
                    )
                })
                .join("\n")
        );
    }
}

/// Create a mock `Body` with a simple flow.
fn goto_switchint<'a>() -> Body<'a> {
    let mut blocks = MockBlocks::new();
    let start = blocks.call(None);
    let goto = blocks.goto(Some(start));
    let switchint = blocks.switchint(Some(goto));
    let then_call = blocks.call(None);
    let else_call = blocks.call(None);
    blocks.set_branch(switchint, 0, then_call);
    blocks.set_branch(switchint, 1, else_call);
    blocks.return_(Some(then_call));
    blocks.return_(Some(else_call));

    let mir_body = blocks.to_body();
    print_mir_graphviz("mir_goto_switchint", &mir_body);
    /* Graphviz character plots created using: `graph-easy --as=boxart`:
                        ┌────────────────┐
                        │   bb0: Call    │
                        └────────────────┘
                          │
                          │
                          ▼
                        ┌────────────────┐
                        │   bb1: Goto    │
                        └────────────────┘
                          │
                          │
                          ▼
    ┌─────────────┐     ┌────────────────┐
    │  bb4: Call  │ ◀── │ bb2: SwitchInt │
    └─────────────┘     └────────────────┘
      │                   │
      │                   │
      ▼                   ▼
    ┌─────────────┐     ┌────────────────┐
    │ bb6: Return │     │   bb3: Call    │
    └─────────────┘     └────────────────┘
                          │
                          │
                          ▼
                        ┌────────────────┐
                        │  bb5: Return   │
                        └────────────────┘
    */
    mir_body
}

#[track_caller]
fn assert_successors(
    graph: &graph::CoverageGraph,
    bcb: BasicCoverageBlock,
    expected_successors: &[BasicCoverageBlock],
) {
    let mut successors = graph.successors[bcb].clone();
    successors.sort_unstable();
    assert_eq!(successors, expected_successors);
}

#[test]
fn test_covgraph_goto_switchint() {
    let mir_body = goto_switchint();
    if false {
        eprintln!("basic_blocks = {}", debug_basic_blocks(&mir_body));
    }
    let graph = graph::CoverageGraph::from_mir(&mir_body);
    print_coverage_graphviz("covgraph_goto_switchint ", &mir_body, &graph);
    /*
    ┌──────────────┐     ┌─────────────────┐
    │ bcb2: Return │ ◀── │ bcb0: SwitchInt │
    └──────────────┘     └─────────────────┘
                           │
                           │
                           ▼
                         ┌─────────────────┐
                         │  bcb1: Return   │
                         └─────────────────┘
    */
    assert_eq!(graph.num_nodes(), 3, "graph: {:?}", graph.iter_enumerated().collect::<Vec<_>>());

    assert_successors(&graph, bcb(0), &[bcb(1), bcb(2)]);
    assert_successors(&graph, bcb(1), &[]);
    assert_successors(&graph, bcb(2), &[]);
}

/// Create a mock `Body` with a loop.
fn switchint_then_loop_else_return<'a>() -> Body<'a> {
    let mut blocks = MockBlocks::new();
    let start = blocks.call(None);
    let switchint = blocks.switchint(Some(start));
    let then_call = blocks.call(None);
    blocks.set_branch(switchint, 0, then_call);
    let backedge_goto = blocks.goto(Some(then_call));
    blocks.link(backedge_goto, switchint);
    let else_return = blocks.return_(None);
    blocks.set_branch(switchint, 1, else_return);

    let mir_body = blocks.to_body();
    print_mir_graphviz("mir_switchint_then_loop_else_return", &mir_body);
    /*
                        ┌────────────────┐
                        │   bb0: Call    │
                        └────────────────┘
                          │
                          │
                          ▼
    ┌─────────────┐     ┌────────────────┐
    │ bb4: Return │ ◀── │ bb1: SwitchInt │ ◀┐
    └─────────────┘     └────────────────┘  │
                          │                 │
                          │                 │
                          ▼                 │
                        ┌────────────────┐  │
                        │   bb2: Call    │  │
                        └────────────────┘  │
                          │                 │
                          │                 │
                          ▼                 │
                        ┌────────────────┐  │
                        │   bb3: Goto    │ ─┘
                        └────────────────┘
    */
    mir_body
}

#[test]
fn test_covgraph_switchint_then_loop_else_return() {
    let mir_body = switchint_then_loop_else_return();
    let graph = graph::CoverageGraph::from_mir(&mir_body);
    print_coverage_graphviz("covgraph_switchint_then_loop_else_return", &mir_body, &graph);
    /*
                       ┌─────────────────┐
                       │   bcb0: Call    │
                       └─────────────────┘
                         │
                         │
                         ▼
    ┌────────────┐     ┌─────────────────┐
    │ bcb3: Goto │ ◀── │ bcb1: SwitchInt │ ◀┐
    └────────────┘     └─────────────────┘  │
      │                  │                  │
      │                  │                  │
      │                  ▼                  │
      │                ┌─────────────────┐  │
      │                │  bcb2: Return   │  │
      │                └─────────────────┘  │
      │                                     │
      └─────────────────────────────────────┘
    */
    assert_eq!(graph.num_nodes(), 4, "graph: {:?}", graph.iter_enumerated().collect::<Vec<_>>());

    assert_successors(&graph, bcb(0), &[bcb(1)]);
    assert_successors(&graph, bcb(1), &[bcb(2), bcb(3)]);
    assert_successors(&graph, bcb(2), &[]);
    assert_successors(&graph, bcb(3), &[bcb(1)]);
}

/// Create a mock `Body` with nested loops.
fn switchint_loop_then_inner_loop_else_break<'a>() -> Body<'a> {
    let mut blocks = MockBlocks::new();
    let start = blocks.call(None);
    let switchint = blocks.switchint(Some(start));
    let then_call = blocks.call(None);
    blocks.set_branch(switchint, 0, then_call);
    let else_return = blocks.return_(None);
    blocks.set_branch(switchint, 1, else_return);

    let inner_start = blocks.call(Some(then_call));
    let inner_switchint = blocks.switchint(Some(inner_start));
    let inner_then_call = blocks.call(None);
    blocks.set_branch(inner_switchint, 0, inner_then_call);
    let inner_backedge_goto = blocks.goto(Some(inner_then_call));
    blocks.link(inner_backedge_goto, inner_switchint);
    let inner_else_break_goto = blocks.goto(None);
    blocks.set_branch(inner_switchint, 1, inner_else_break_goto);

    let backedge_goto = blocks.goto(Some(inner_else_break_goto));
    blocks.link(backedge_goto, switchint);

    let mir_body = blocks.to_body();
    print_mir_graphviz("mir_switchint_loop_then_inner_loop_else_break", &mir_body);
    /*
                        ┌────────────────┐
                        │   bb0: Call    │
                        └────────────────┘
                          │
                          │
                          ▼
    ┌─────────────┐     ┌────────────────┐
    │ bb3: Return │ ◀── │ bb1: SwitchInt │ ◀─────┐
    └─────────────┘     └────────────────┘       │
                          │                      │
                          │                      │
                          ▼                      │
                        ┌────────────────┐       │
                        │   bb2: Call    │       │
                        └────────────────┘       │
                          │                      │
                          │                      │
                          ▼                      │
                        ┌────────────────┐       │
                        │   bb4: Call    │       │
                        └────────────────┘       │
                          │                      │
                          │                      │
                          ▼                      │
    ┌─────────────┐     ┌────────────────┐       │
    │  bb8: Goto  │ ◀── │ bb5: SwitchInt │ ◀┐    │
    └─────────────┘     └────────────────┘  │    │
      │                   │                 │    │
      │                   │                 │    │
      ▼                   ▼                 │    │
    ┌─────────────┐     ┌────────────────┐  │    │
    │  bb9: Goto  │ ─┐  │   bb6: Call    │  │    │
    └─────────────┘  │  └────────────────┘  │    │
                     │    │                 │    │
                     │    │                 │    │
                     │    ▼                 │    │
                     │  ┌────────────────┐  │    │
                     │  │   bb7: Goto    │ ─┘    │
                     │  └────────────────┘       │
                     │                           │
                     └───────────────────────────┘
    */
    mir_body
}

#[test]
fn test_covgraph_switchint_loop_then_inner_loop_else_break() {
    let mir_body = switchint_loop_then_inner_loop_else_break();
    let graph = graph::CoverageGraph::from_mir(&mir_body);
    print_coverage_graphviz(
        "covgraph_switchint_loop_then_inner_loop_else_break",
        &mir_body,
        &graph,
    );
    /*
                         ┌─────────────────┐
                         │   bcb0: Call    │
                         └─────────────────┘
                           │
                           │
                           ▼
    ┌──────────────┐     ┌─────────────────┐
    │ bcb2: Return │ ◀── │ bcb1: SwitchInt │ ◀┐
    └──────────────┘     └─────────────────┘  │
                           │                  │
                           │                  │
                           ▼                  │
                         ┌─────────────────┐  │
                         │   bcb3: Call    │  │
                         └─────────────────┘  │
                           │                  │
                           │                  │
                           ▼                  │
    ┌──────────────┐     ┌─────────────────┐  │
    │  bcb6: Goto  │ ◀── │ bcb4: SwitchInt │ ◀┼────┐
    └──────────────┘     └─────────────────┘  │    │
      │                    │                  │    │
      │                    │                  │    │
      │                    ▼                  │    │
      │                  ┌─────────────────┐  │    │
      │                  │   bcb5: Goto    │ ─┘    │
      │                  └─────────────────┘       │
      │                                            │
      └────────────────────────────────────────────┘
    */
    assert_eq!(graph.num_nodes(), 7, "graph: {:?}", graph.iter_enumerated().collect::<Vec<_>>());

    assert_successors(&graph, bcb(0), &[bcb(1)]);
    assert_successors(&graph, bcb(1), &[bcb(2), bcb(3)]);
    assert_successors(&graph, bcb(2), &[]);
    assert_successors(&graph, bcb(3), &[bcb(4)]);
    assert_successors(&graph, bcb(4), &[bcb(5), bcb(6)]);
    assert_successors(&graph, bcb(5), &[bcb(1)]);
    assert_successors(&graph, bcb(6), &[bcb(4)]);
}
