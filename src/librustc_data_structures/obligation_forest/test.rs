// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::iter::FromIterator;
use std::rc::Rc;

use super::{ObligationForest, UndoLog, Outcome, Error};
use super::super::undoable::{Undoable, UndoableTracker, Undoer};


#[derive(Debug)]
pub struct TestTreeStateUndoer { old_value: usize }
impl Undoer for TestTreeStateUndoer {
    type Undoable = TestTreeState;
    fn undo(self, state: &mut TestTreeState) {
        state.value = self.old_value;
    }
}

#[derive(Debug)]
pub struct TestTreeState {
    id: Option<usize>,
    tracker: Option<Rc<RefCell<UndoLog<&'static str, TestTreeState>>>>,
    value: usize,
}
impl Undoable for TestTreeState {
    type Undoer = TestTreeStateUndoer;
    type Tracker = UndoLog<&'static str, TestTreeState>;
    fn register_tracker(
        &mut self, tracker: Rc<RefCell<UndoLog<&'static str, TestTreeState>>>, id: usize)
    {
        self.id = Some(id);
        self.tracker = Some(tracker);
    }
}
impl TestTreeState {
    fn new(value: usize) -> TestTreeState {
        TestTreeState { id: None, tracker: None, value: value }
    }
    fn set(&mut self, value: usize) {
        let mut tracker = self.tracker.as_ref().unwrap().borrow_mut();
        tracker.push_action(self.id.unwrap(), TestTreeStateUndoer { old_value: self.value });
        self.value = value;
    }
}

#[test]
fn push_snap_push_snap_modify_commit_rollback() {
    let mut forest = ObligationForest::new();
    forest.push_tree("A", TestTreeState::new(0));
    forest.push_tree("B", TestTreeState::new(1));
    forest.push_tree("C", TestTreeState::new(2));

    let snap0 = forest.start_snapshot();
    forest.push_tree("D", TestTreeState::new(3));

    let snap1 = forest.start_snapshot();
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            match *obligation {
                "A" => {
                    assert_eq!(tree.value, 0);
                    tree.set(4);
                    Ok(Some(vec!["A.1", "A.2", "A.3"]))
                },
                "B" => {
                    assert_eq!(tree.value, 1);
                    tree.set(5);
                    Err("B is for broken")
                },
                "C" => {
                    assert_eq!(tree.value, 2);
                    tree.set(6);
                    Ok(Some(vec![]))
                },
                "D" => {
                    assert_eq!(tree.value, 3);
                    Ok(Some(vec!["D.1"]))
                }
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error { error: "B is for broken",
                                 backtrace: vec!["B"] }]);

    forest.commit_snapshot(snap1);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            match *obligation {
                "A.1" => {
                    assert_eq!(tree.value, 4);
                    Ok(Some(vec![]))
                },
                "A.2" => {
                    assert_eq!(tree.value, 4);
                    Err("A.2 broke too")
                },
                "A.3" => {
                    assert_eq!(tree.value, 4);
                    Ok(Some(vec!["A.3.1"]))
                },
                "D.1" => {
                    assert_eq!(tree.value, 3);
                    Ok(Some(vec![]))
                }
                _ => unreachable!(),
            }
        });
    assert_eq!(BTreeSet::from_iter(ok), BTreeSet::from_iter(vec!["A.1", "D.1", "D"]));
    assert_eq!(err, vec![Error { error: "A.2 broke too",
                                 backtrace: vec!["A.2", "A"] }]);

    forest.rollback_snapshot(snap0);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            match *obligation {
                "A" => {
                    assert_eq!(tree.value, 0);
                    Ok(Some(vec!["A.1", "A.2", "A.3"]))
                },
                "B" => {
                    assert_eq!(tree.value, 1);
                    Err("B is for broken")
                },
                "C" => {
                    assert_eq!(tree.value, 2);
                    Ok(Some(vec![]))
                },
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error { error: "B is for broken",
                                 backtrace: vec!["B"] }]);
}

#[test]
fn push_snap_modify_rollback() {
    let mut forest = ObligationForest::new();
    forest.push_tree("A", TestTreeState::new(0));
    forest.push_tree("B", TestTreeState::new(1));
    forest.push_tree("C", TestTreeState::new(2));

    let snap0 = forest.start_snapshot();
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            match *obligation {
                "A" => {
                    assert_eq!(tree.value, 0);
                    tree.set(3);
                    Ok(Some(vec!["A.1", "A.2", "A.3"]))
                },
                "B" => {
                    assert_eq!(tree.value, 1);
                    tree.set(4);
                    Err("B is for broken")
                },
                "C" => {
                    assert_eq!(tree.value, 2);
                    tree.set(5);
                    Ok(Some(vec![]))
                },
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error { error: "B is for broken",
                                 backtrace: vec!["B"] }]);

    let _ =
        forest.process_obligations(|obligation, tree, _| {
            match *obligation {
                "A.1" => {
                    assert_eq!(tree.value, 3);
                    Ok(None)
                },
                "A.2" => {
                    assert_eq!(tree.value, 3);
                    Err("A.2 broke too")
                },
                "A.3" => {
                    assert_eq!(tree.value, 3);
                    Ok(Some(vec!["A.3.1"]))
                },
                _ => unreachable!(),
            }
        });

    forest.rollback_snapshot(snap0);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            match *obligation {
                "A" => {
                    assert_eq!(tree.value, 0);
                    Ok(Some(vec!["A.1", "A.2", "A.3"]))
                },
                "B" => {
                    assert_eq!(tree.value, 1);
                    Err("B is for broken")
                },
                "C" => {
                    assert_eq!(tree.value, 2);
                    Ok(Some(vec![]))
                },
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error { error: "B is for broken",
                                 backtrace: vec!["B"] }]);
}

#[derive(Debug)]
pub struct StrUndoer;
impl Undoer for StrUndoer {
    type Undoable = &'static str;
    fn undo(self, _: &mut &'static str) {}
}

impl Undoable for &'static str {
    type Undoer = StrUndoer;
    type Tracker = UndoLog<&'static str, &'static str>;
    fn register_tracker(&mut self, _: Rc<RefCell<Self::Tracker>>, _: usize) {
        // noop
    }
}

#[test]
fn push_pop() {
    let mut forest = ObligationForest::new();
    forest.push_tree("A", "A");
    forest.push_tree("B", "B");
    forest.push_tree("C", "C");

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                "B" => Err("B is for broken"),
                "C" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error {error: "B is for broken",
                                backtrace: vec!["B"]}]);

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.push_tree("D", "D");
    let Outcome { completed: ok, errors: err, .. }: Outcome<&'static str, ()> =
        forest.process_obligations(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A.1" => Ok(None),
                "A.2" => Ok(None),
                "A.3" => Ok(Some(vec!["A.3.i"])),
                "D" => Ok(Some(vec!["D.1", "D.2"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, Vec::new());


    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A.3.i, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A.1" => Ok(Some(vec![])),
                "A.2" => Err("A is for apple"),
                "D.1" => Ok(Some(vec!["D.1.i"])),
                "D.2" => Ok(Some(vec!["D.2.i"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["A.1"]);
    assert_eq!(err, vec![Error { error: "A is for apple",
                                 backtrace: vec!["A.2", "A"] }]);

    // fourth round: error in D.1.i that should propagate to D.2.i
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "D.1.i" => Err("D is for dumb"),
                _ => panic!("unexpected obligation {:?}", obligation),
            }
        });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, vec![Error { error: "D is for dumb",
                                 backtrace: vec!["D.1.i", "D.1", "D"] }]);
}

// Test that if a tree with grandchildren succeeds, everything is
// reported as expected:
// A
//   A.1
//   A.2
//      A.2.i
//      A.2.ii
//   A.3
#[test]
fn success_in_grandchildren() {
    let mut forest = ObligationForest::new();
    forest.push_tree("A", "A");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A.1" => Ok(Some(vec![])),
                "A.2" => Ok(Some(vec!["A.2.i", "A.2.ii"])),
                "A.3" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(BTreeSet::from_iter(ok), BTreeSet::from_iter(vec!["A.3", "A.1"]));
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A.2.i" => Ok(Some(vec!["A.2.i.a"])),
                "A.2.ii" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["A.2.ii"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A.2.i.a" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["A.2.i.a", "A.2.i", "A.2", "A"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|_, _, _| unreachable!());
    assert!(ok.is_empty());
    assert!(err.is_empty());
}

#[test]
fn to_errors_no_throw() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest = ObligationForest::new();
    forest.push_tree("A", "A");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, _| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);
    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 1);
}

#[test]
fn backtrace() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest = ObligationForest::new();
    forest.push_tree("A", "A");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, mut backtrace| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            assert!(backtrace.next().is_none());
            match *obligation {
                "A" => Ok(Some(vec!["A.1"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, mut backtrace| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1" => Ok(Some(vec!["A.1.i"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, tree, mut backtrace| {
            assert_eq!(obligation.chars().next(), tree.chars().next());
            assert!(backtrace.next().unwrap() == &"A.1");
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1.i" => Ok(None),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert!(err.is_empty());
}
