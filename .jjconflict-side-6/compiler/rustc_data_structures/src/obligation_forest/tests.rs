use std::fmt;

use thin_vec::thin_vec;

use super::*;

impl<'a> super::ForestObligation for &'a str {
    type CacheKey = &'a str;

    fn as_cache_key(&self) -> Self::CacheKey {
        self
    }
}

struct ClosureObligationProcessor<OF, BF, O, E> {
    process_obligation: OF,
    _process_backedge: BF,
    marker: PhantomData<(O, E)>,
}

struct TestOutcome<O, E> {
    pub completed: Vec<O>,
    pub errors: Vec<Error<O, E>>,
}

impl<O, E> OutcomeTrait for TestOutcome<O, E>
where
    O: Clone,
{
    type Error = Error<O, E>;
    type Obligation = O;

    fn new() -> Self {
        Self { errors: vec![], completed: vec![] }
    }

    fn record_completed(&mut self, outcome: &Self::Obligation) {
        self.completed.push(outcome.clone())
    }

    fn record_error(&mut self, error: Self::Error) {
        self.errors.push(error)
    }
}

#[allow(non_snake_case)]
fn C<OF, BF, O>(of: OF, bf: BF) -> ClosureObligationProcessor<OF, BF, O, &'static str>
where
    OF: FnMut(&mut O) -> ProcessResult<O, &'static str>,
    BF: FnMut(&[O]),
{
    ClosureObligationProcessor {
        process_obligation: of,
        _process_backedge: bf,
        marker: PhantomData,
    }
}

impl<OF, BF, O, E> ObligationProcessor for ClosureObligationProcessor<OF, BF, O, E>
where
    O: super::ForestObligation + fmt::Debug,
    E: fmt::Debug,
    OF: FnMut(&mut O) -> ProcessResult<O, E>,
    BF: FnMut(&[O]),
{
    type Obligation = O;
    type Error = E;
    type OUT = TestOutcome<O, E>;

    fn needs_process_obligation(&self, _obligation: &Self::Obligation) -> bool {
        true
    }

    fn process_obligation(
        &mut self,
        obligation: &mut Self::Obligation,
    ) -> ProcessResult<Self::Obligation, Self::Error> {
        (self.process_obligation)(obligation)
    }

    fn process_backedge<'c, I>(
        &mut self,
        _cycle: I,
        _marker: PhantomData<&'c Self::Obligation>,
    ) -> Result<(), Self::Error>
    where
        I: Clone + Iterator<Item = &'c Self::Obligation>,
    {
        Ok(())
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
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Changed(thin_vec!["A.1", "A.2", "A.3"]),
            "B" => ProcessResult::Error("B is for broken"),
            "C" => ProcessResult::Changed(thin_vec![]),
            "A.1" | "A.2" | "A.3" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error { error: "B is for broken", backtrace: vec!["B"] }]);

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.register_obligation("D");
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A.1" => ProcessResult::Unchanged,
            "A.2" => ProcessResult::Unchanged,
            "A.3" => ProcessResult::Changed(thin_vec!["A.3.i"]),
            "D" => ProcessResult::Changed(thin_vec!["D.1", "D.2"]),
            "A.3.i" | "D.1" | "D.2" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, Vec::new());

    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A.1" => ProcessResult::Changed(thin_vec![]),
            "A.2" => ProcessResult::Error("A is for apple"),
            "A.3.i" => ProcessResult::Changed(thin_vec![]),
            "D.1" => ProcessResult::Changed(thin_vec!["D.1.i"]),
            "D.2" => ProcessResult::Changed(thin_vec!["D.2.i"]),
            "D.1.i" | "D.2.i" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["A.1", "A.3", "A.3.i"]);
    assert_eq!(err, vec![Error { error: "A is for apple", backtrace: vec!["A.2", "A"] }]);

    // fourth round: error in D.1.i
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "D.1.i" => ProcessResult::Error("D is for dumb"),
            "D.2.i" => ProcessResult::Changed(thin_vec![]),
            _ => panic!("unexpected obligation {:?}", obligation),
        },
        |_| {},
    ));
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["D.2", "D.2.i"]);
    assert_eq!(err, vec![Error { error: "D is for dumb", backtrace: vec!["D.1.i", "D.1", "D"] }]);
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

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Changed(thin_vec!["A.1", "A.2", "A.3"]),
            "A.1" => ProcessResult::Changed(thin_vec![]),
            "A.2" => ProcessResult::Changed(thin_vec!["A.2.i", "A.2.ii"]),
            "A.3" => ProcessResult::Changed(thin_vec![]),
            "A.2.i" | "A.2.ii" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["A.1", "A.3"]);
    assert!(err.is_empty());

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A.2.i" => ProcessResult::Unchanged,
            "A.2.ii" => ProcessResult::Changed(thin_vec![]),
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok, vec!["A.2.ii"]);
    assert!(err.is_empty());

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A.2.i" => ProcessResult::Changed(thin_vec!["A.2.i.a"]),
            "A.2.i.a" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert!(ok.is_empty());
    assert!(err.is_empty());

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A.2.i.a" => ProcessResult::Changed(thin_vec![]),
            _ => unreachable!(),
        },
        |_| {},
    ));
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["A", "A.2", "A.2.i", "A.2.i.a"]);
    assert!(err.is_empty());

    let TestOutcome { completed: ok, errors: err, .. } =
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
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Changed(thin_vec!["A.1", "A.2", "A.3"]),
            "A.1" | "A.2" | "A.3" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
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
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Changed(thin_vec!["A.1", "A.2"]),
            "A.1" | "A.2" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A.1" => ProcessResult::Changed(thin_vec!["D"]),
            "A.2" => ProcessResult::Changed(thin_vec!["D"]),
            "D" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let mut d_count = 0;
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "D" => {
                d_count += 1;
                ProcessResult::Changed(thin_vec![])
            }
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(d_count, 1);
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["A", "A.1", "A.2", "D"]);
    assert_eq!(err.len(), 0);

    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 0);

    forest.register_obligation("A'");
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A'" => ProcessResult::Changed(thin_vec!["A'.1", "A'.2"]),
            "A'.1" | "A'.2" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A'.1" => ProcessResult::Changed(thin_vec!["D'", "A'"]),
            "A'.2" => ProcessResult::Changed(thin_vec!["D'"]),
            "D'" | "A'" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let mut d_count = 0;
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "D'" => {
                d_count += 1;
                ProcessResult::Error("operation failed")
            }
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(d_count, 1);
    assert_eq!(ok.len(), 0);
    assert_eq!(
        err,
        vec![super::Error { error: "operation failed", backtrace: vec!["D'", "A'.1", "A'"] }]
    );

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

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A: Sized" | "B: Sized" | "C: Sized" => ProcessResult::Changed(thin_vec![]),
            _ => unreachable!(),
        },
        |_| {},
    ));
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["A: Sized", "B: Sized", "C: Sized"]);
    assert_eq!(err.len(), 0);

    forest.register_obligation("(A,B,C): Sized");
    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "(A,B,C): Sized" => {
                ProcessResult::Changed(thin_vec!["A: Sized", "B: Sized", "C: Sized"])
            }
            _ => unreachable!(),
        },
        |_| {},
    ));
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

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Changed(thin_vec!["D", "E"]),
            "B" => ProcessResult::Unchanged,
            "C1" => ProcessResult::Changed(thin_vec![]),
            "C2" => ProcessResult::Changed(thin_vec![]),
            "D" | "E" => ProcessResult::Unchanged,
            _ => unreachable!(),
        },
        |_| {},
    ));
    let mut ok = ok;
    ok.sort();
    assert_eq!(ok, vec!["C1", "C2"]);
    assert_eq!(err.len(), 0);

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "D" | "E" => ProcessResult::Unchanged,
            "B" => ProcessResult::Changed(thin_vec!["D"]),
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "D" => ProcessResult::Unchanged,
            "E" => ProcessResult::Error("E is for error"),
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error { error: "E is for error", backtrace: vec!["E", "A"] }]);

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "D" => ProcessResult::Error("D is dead"),
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error { error: "D is dead", backtrace: vec!["D"] }]);

    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 0);
}

#[test]
fn simultaneous_register_and_error() {
    // check that registering a failed obligation works correctly
    let mut forest = ObligationForest::new();
    forest.register_obligation("A");
    forest.register_obligation("B");

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Error("An error"),
            "B" => ProcessResult::Changed(thin_vec!["A"]),
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error { error: "An error", backtrace: vec!["A"] }]);

    let mut forest = ObligationForest::new();
    forest.register_obligation("B");
    forest.register_obligation("A");

    let TestOutcome { completed: ok, errors: err, .. } = forest.process_obligations(&mut C(
        |obligation| match *obligation {
            "A" => ProcessResult::Error("An error"),
            "B" => ProcessResult::Changed(thin_vec!["A"]),
            _ => unreachable!(),
        },
        |_| {},
    ));
    assert_eq!(ok.len(), 0);
    assert_eq!(err, vec![super::Error { error: "An error", backtrace: vec!["A"] }]);
}
