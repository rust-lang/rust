use super::{ObligationForest, Outcome, Error};

#[test]
fn push_pop() {
    let mut forest = ObligationForest::new();
    forest.push_root("A");
    forest.push_root("B");
    forest.push_root("C");

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
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
    forest.push_root("D");
    let Outcome { successful: ok, errors: err, .. }: Outcome<&'static str, ()> =
        forest.process_obligations(|obligation, _| {
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
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
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
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
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
    forest.push_root("A");

    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, _| {
        match *obligation {
            "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
            _ => unreachable!(),
        }
    });
    assert!(ok.is_empty());
    assert!(err.is_empty());

    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, _| {
        match *obligation {
            "A.1" => Ok(Some(vec![])),
            "A.2" => Ok(Some(vec!["A.2.i", "A.2.ii"])),
            "A.3" => Ok(Some(vec![])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A.3", "A.1"]);
    assert!(err.is_empty());

    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, _| {
        match *obligation {
            "A.2.i" => Ok(Some(vec!["A.2.i.a"])),
            "A.2.ii" => Ok(Some(vec![])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A.2.ii"]);
    assert!(err.is_empty());

    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, _| {
        match *obligation {
            "A.2.i.a" => Ok(Some(vec![])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A.2.i.a", "A.2.i", "A.2", "A"]);
    assert!(err.is_empty());

    let Outcome { successful: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|_, _| unreachable!());
    assert!(ok.is_empty());
    assert!(err.is_empty());
}

#[test]
fn to_errors_no_throw() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest = ObligationForest::new();
    forest.push_root("A");
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, _| {
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
    let mut forest: ObligationForest<&'static str> = ObligationForest::new();
    forest.push_root("A");
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, mut backtrace| {
        assert!(backtrace.next().is_none());
        match *obligation {
            "A" => Ok(Some(vec!["A.1"])),
            _ => unreachable!(),
        }
    });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, mut backtrace| {
        assert!(backtrace.next().unwrap() == &"A");
        assert!(backtrace.next().is_none());
        match *obligation {
            "A.1" => Ok(Some(vec!["A.1.i"])),
            _ => unreachable!(),
        }
    });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { successful: ok, errors: err, .. } = forest.process_obligations::<(),_>(|obligation, mut backtrace| {
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
