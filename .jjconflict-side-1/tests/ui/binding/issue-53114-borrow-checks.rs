//@ check-pass
// Issue #53114: NLL's borrow check had some deviations from the old borrow
// checker, and both had some deviations from our ideal state. This test
// captures the behavior of how `_` bindings are handled with respect to how we
// flag expressions that are meant to request unsafe blocks.
#![allow(irrefutable_let_patterns, dropping_references)]
struct M;

fn let_wild_gets_moved_expr() {
    let m = M;
    drop(m);
    let _ = m; // accepted, and want it to continue to be

    let mm = (M, M); // variation on above with `_` in substructure
    let (_x, _) = mm;
    let (_, _y) = mm;
    let (_, _) = mm;
}

fn match_moved_expr_to_wild() {
    let m = M;
    drop(m);
    match m { _ => { } } // #53114: accepted too

    let mm = (M, M); // variation on above with `_` in substructure
    match mm { (_x, _) => { } }
    match mm { (_, _y) => { } }
    match mm { (_, _) => { } }
}

fn if_let_moved_expr_to_wild() {
    let m = M;
    drop(m);
    if let _ = m { } // #53114: accepted too

    let mm = (M, M); // variation on above with `_` in substructure
    if let (_x, _) = mm { }
    if let (_, _y) = mm { }
    if let (_, _) = mm { }
}

fn let_wild_gets_borrowed_expr() {
    let mut m = M;
    let r = &mut m;
    let _ = m; // accepted, and want it to continue to be
    // let _x = m; // (compare with this error.)
    drop(r);

    let mut mm = (M, M); // variation on above with `_` in substructure
    let (r1, r2) = (&mut mm.0, &mut mm.1);
    let (_, _) = mm;
    drop((r1, r2));
}

fn match_borrowed_expr_to_wild() {
    let mut m = M;
    let r = &mut m;
    match m { _ => {} } ; // accepted, and want it to continue to be
    drop(r);

    let mut mm = (M, M); // variation on above with `_` in substructure
    let (r1, r2) = (&mut mm.0, &mut mm.1);
    match mm { (_, _) => { } }
    drop((r1, r2));
}

fn if_let_borrowed_expr_to_wild() {
    let mut m = M;
    let r = &mut m;
    if let _ = m { } // accepted, and want it to continue to be
    drop(r);

    let mut mm = (M, M); // variation on above with `_` in substructure
    let (r1, r2) = (&mut mm.0, &mut mm.1);
    if let (_, _) = mm { }
    drop((r1, r2));
}

fn main() { }
