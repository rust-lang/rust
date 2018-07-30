// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg(test)]

use super::{Error, ObligationForest, ObligationProcessor, Outcome, ProcessResult};

use std::fmt;
use std::marker::PhantomData;

impl<'a> super::ForestObligation for &'a str {
    type Predicate = &'a str;

    fn as_predicate(&self) -> &Self::Predicate {
        self
    }
}

struct ClosureObligationProcessor<OF, BF, O, E> {
    process_obligation: OF,
    _process_backedge: BF,
    marker: PhantomData<(O, E)>,
}

#[allow(non_snake_case)]
fn C<OF, BF, O>(of: OF, bf: BF) -> ClosureObligationProcessor<OF, BF, O, &'static str>
    where OF: FnMut(&mut O) -> ProcessResult<O, &'static str>,
          BF: FnMut(&[O])
{
    ClosureObligationProcessor {
        process_obligation: of,
        _process_backedge: bf,
        marker: PhantomData
    }
}

impl<OF, BF, O, E> ObligationProcessor for ClosureObligationProcessor<OF, BF, O, E>
    where O: super::ForestObligation + fmt::Debug,
          E: fmt::Debug,
          OF: FnMut(&mut O) -> ProcessResult<O, E>,
          BF: FnMut(&[O])
{
    type Obligation = O;
    type Error = E;

    fn process_obligation(&mut self,
                          obligation: &mut Self::Obligation)
                          -> ProcessResult<Self::Obligation, Self::Error>
    {
        (self.process_obligation)(obligation)
    }

    fn process_backedge<'c, I>(&mut self, _cycle: I,
                               _marker: PhantomData<&'c Self::Obligation>)
        where I: Clone + Iterator<Item=&'c Self::Obligation> {
        }
}


#[test]
fn push_pop() {
    let mut forest = ObligationForest::new();
    forest.register_obligation("A");
    forest.register_obligation("B");
    forest.register_obligation("C");

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Changed(vec!["A.1", "A.2", "A.3"]),
                "B" => ProcessResult::Error("B is for broken"),
                "C" => ProcessResult::Changed(vec![]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err,
               vec![Error {
                        error: "B is for broken",
                        backtrace: vec!["B"],
                    }]);

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.register_obligation("D");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A.1" => ProcessResult::Unchanged,
                "A.2" => ProcessResult::Unchanged,
                "A.3" => ProcessResult::Changed(vec!["A.3.i"]),
                "D" => ProcessResult::Changed(vec!["D.1", "D.2"]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, Vec::new());


    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A.1" => ProcessResult::Changed(vec![]),
                "A.2" => ProcessResult::Error("A is for apple"),
                "A.3.i" => ProcessResult::Changed(vec![]),
                "D.1" => ProcessResult::Changed(vec!["D.1.i"]),
                "D.2" => ProcessResult::Changed(vec!["D.2.i"]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert_eq!(ok, vec!["A.3", "A.1", "A.3.i"]);
    assert_eq!(err,
               vec![Error {
                        error: "A is for apple",
                        backtrace: vec!["A.2", "A"],
                    }]);

    // fourth round: error in D.1.i
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "D.1.i" => ProcessResult::Error("D is for dumb"),
                "D.2.i" => ProcessResult::Changed(vec![]),
                _ => panic!("unexpected obligation {:?}", obligation),
            }
        }, |_| {}));
    assert_eq!(ok, vec!["D.2.i", "D.2"]);
    assert_eq!(err,
               vec![Error {
                        error: "D is for dumb",
                        backtrace: vec!["D.1.i", "D.1", "D"],
                    }]);
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
    forest.register_obligation("A");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Changed(vec!["A.1", "A.2", "A.3"]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert!(ok.is_empty());
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A.1" => ProcessResult::Changed(vec![]),
                "A.2" => ProcessResult::Changed(vec!["A.2.i", "A.2.ii"]),
                "A.3" => ProcessResult::Changed(vec![]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert_eq!(ok, vec!["A.3", "A.1"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A.2.i" => ProcessResult::Changed(vec!["A.2.i.a"]),
                "A.2.ii" => ProcessResult::Changed(vec![]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert_eq!(ok, vec!["A.2.ii"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A.2.i.a" => ProcessResult::Changed(vec![]),
                _ => unreachable!(),
            }
        }, |_| {}));
    assert_eq!(ok, vec!["A.2.i.a", "A.2.i", "A.2", "A"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|_| unreachable!(), |_| {}));

    assert!(ok.is_empty());
    assert!(err.is_empty());
}

#[test]
fn to_errors_no_throw() {
    // check that converting multiple children with common parent (A)
    // yields to correct errors (and does not panic, in particular).
    let mut forest = ObligationForest::new();
    forest.register_obligation("A");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Changed(vec!["A.1", "A.2", "A.3"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);
    let errors = forest.to_errors(());
    assert_eq!(errors[0].backtrace, vec!["A.1", "A"]);
    assert_eq!(errors[1].backtrace, vec!["A.2", "A"]);
    assert_eq!(errors[2].backtrace, vec!["A.3", "A"]);
    assert_eq!(errors.len(), 3);
}

#[test]
fn diamond() {
    // check that diamond dependencies are handled correctly
    let mut forest = ObligationForest::new();
    forest.register_obligation("A");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Changed(vec!["A.1", "A.2"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A.1" => ProcessResult::Changed(vec!["D"]),
                "A.2" => ProcessResult::Changed(vec!["D"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let mut d_count = 0;
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "D" => { d_count += 1; ProcessResult::Changed(vec![]) },
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(d_count, 1);
    assert_eq!(ok, vec!["D", "A.2", "A.1", "A"]);
    assert_eq!(err.len(), 0);

    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 0);

    forest.register_obligation("A'");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A'" => ProcessResult::Changed(vec!["A'.1", "A'.2"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A'.1" => ProcessResult::Changed(vec!["D'", "A'"]),
                "A'.2" => ProcessResult::Changed(vec!["D'"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let mut d_count = 0;
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "D'" => { d_count += 1; ProcessResult::Error("operation failed") },
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(d_count, 1);
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error {
        error: "operation failed",
        backtrace: vec!["D'", "A'.1", "A'"]
    }]);

    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 0);
}

#[test]
fn done_dependency() {
    // check that the local cache works
    let mut forest = ObligationForest::new();
    forest.register_obligation("A: Sized");
    forest.register_obligation("B: Sized");
    forest.register_obligation("C: Sized");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A: Sized" | "B: Sized" | "C: Sized" => ProcessResult::Changed(vec![]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok, vec!["C: Sized", "B: Sized", "A: Sized"]);
    assert_eq!(err.len(), 0);

    forest.register_obligation("(A,B,C): Sized");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "(A,B,C): Sized" => ProcessResult::Changed(vec![
                    "A: Sized",
                    "B: Sized",
                    "C: Sized"
                        ]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok, vec!["(A,B,C): Sized"]);
    assert_eq!(err.len(), 0);


}


#[test]
fn orphan() {
    // check that orphaned nodes are handled correctly
    let mut forest = ObligationForest::new();
    forest.register_obligation("A");
    forest.register_obligation("B");
    forest.register_obligation("C1");
    forest.register_obligation("C2");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Changed(vec!["D", "E"]),
                "B" => ProcessResult::Unchanged,
                "C1" => ProcessResult::Changed(vec![]),
                "C2" => ProcessResult::Changed(vec![]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok, vec!["C2", "C1"]);
    assert_eq!(err.len(), 0);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "D" | "E" => ProcessResult::Unchanged,
                "B" => ProcessResult::Changed(vec!["D"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "D" => ProcessResult::Unchanged,
                "E" => ProcessResult::Error("E is for error"),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error {
        error: "E is for error",
        backtrace: vec!["E", "A"]
    }]);

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "D" => ProcessResult::Error("D is dead"),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error {
        error: "D is dead",
        backtrace: vec!["D"]
    }]);

    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 0);
}

#[test]
fn simultaneous_register_and_error() {
    // check that registering a failed obligation works correctly
    let mut forest = ObligationForest::new();
    forest.register_obligation("A");
    forest.register_obligation("B");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Error("An error"),
                "B" => ProcessResult::Changed(vec!["A"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error {
        error: "An error",
        backtrace: vec!["A"]
    }]);

    let mut forest = ObligationForest::new();
    forest.register_obligation("B");
    forest.register_obligation("A");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations(&mut C(|obligation| {
            match *obligation {
                "A" => ProcessResult::Error("An error"),
                "B" => ProcessResult::Changed(vec!["A"]),
                _ => unreachable!(),
            }
        }, |_|{}));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error {
        error: "An error",
        backtrace: vec!["A"]
    }]);
}
