
import tritv::*;

type precond = t;

/* 2 means "this constraint may or may not be true after execution"
   1 means "this constraint is definitely true after execution"
   0 means "this constraint is definitely false after execution" */
type postcond = t;


/* 2 means "don't know about this constraint"
   1 means "this constraint is definitely true before entry"
   0 means "this constraint is definitely false on entry" */
type prestate = t;


/* similar to postcond */
type poststate = t;


/* 1 means "this variable is definitely initialized"
  0 means "don't know whether this variable is
  initialized" */

/*
   This says: this expression requires the constraints whose value is 1 in
   <pre> to be true, and given the precondition, it guarantees that the
   constraints in <post> whose values are 1 are true, and that the constraints
   in <post> whose values are 0 are false.
 */

/* named thus so as not to confuse with prestate and poststate */
type pre_and_post = {precondition: precond, postcondition: postcond};


/* FIXME: once it's implemented: (Issue #34) */

//  : ((*.precondition).nbits == (*.postcondition).nbits);
type pre_and_post_state = {prestate: prestate, poststate: poststate};

type ts_ann = {conditions: pre_and_post, states: pre_and_post_state};

fn true_precond(num_vars: uint) -> precond { return create_tritv(num_vars); }

fn true_postcond(num_vars: uint) -> postcond {
    return true_precond(num_vars);
}

fn empty_prestate(num_vars: uint) -> prestate {
    return true_precond(num_vars);
}

fn empty_poststate(num_vars: uint) -> poststate {
    return true_precond(num_vars);
}

fn false_postcond(num_vars: uint) -> postcond {
    let rslt = create_tritv(num_vars);
    rslt.set_all();
    rslt
}

fn empty_pre_post(num_vars: uint) -> pre_and_post {
    return {precondition: empty_prestate(num_vars),
         postcondition: empty_poststate(num_vars)};
}

fn empty_states(num_vars: uint) -> pre_and_post_state {
    return {prestate: true_precond(num_vars),
         poststate: true_postcond(num_vars)};
}

fn empty_ann(num_vars: uint) -> ts_ann {
    return {conditions: empty_pre_post(num_vars),
         states: empty_states(num_vars)};
}

fn get_pre(&&p: pre_and_post) -> precond { return p.precondition; }

fn get_post(&&p: pre_and_post) -> postcond { return p.postcondition; }

fn difference(p1: precond, p2: precond) -> bool { p1.difference(p2) }

fn union(p1: precond, p2: precond) -> bool { p1.union(p2) }

fn intersect(p1: precond, p2: precond) -> bool { p1.intersect(p2) }

fn pps_len(p: pre_and_post) -> uint {
    // gratuitous check

    assert (p.precondition.nbits == p.postcondition.nbits);
    return p.precondition.nbits;
}

fn require(i: uint, p: pre_and_post) {
    // sets the ith bit in p's pre
    p.precondition.set(i, ttrue);
}

fn require_and_preserve(i: uint, p: pre_and_post) {
    // sets the ith bit in p's pre and post
    p.precondition.set(i, ttrue);
    p.postcondition.set(i, ttrue);
}

fn set_in_postcond(i: uint, p: pre_and_post) -> bool {
    // sets the ith bit in p's post
    return set_in_postcond_(i, p.postcondition);
}

fn set_in_postcond_(i: uint, p: postcond) -> bool {
    let was_set = p.get(i);
    p.set(i, ttrue);
    return was_set != ttrue;
}

fn set_in_poststate(i: uint, s: pre_and_post_state) -> bool {
    // sets the ith bit in p's post
    return set_in_poststate_(i, s.poststate);
}

fn set_in_poststate_(i: uint, p: poststate) -> bool {
    let was_set = p.get(i);
    p.set(i, ttrue);
    return was_set != ttrue;

}

fn clear_in_poststate(i: uint, s: pre_and_post_state) -> bool {
    // sets the ith bit in p's post
    return clear_in_poststate_(i, s.poststate);
}

fn clear_in_poststate_(i: uint, s: poststate) -> bool {
    let was_set = s.get(i);
    s.set(i, tfalse);
    return was_set != tfalse;
}

fn clear_in_prestate(i: uint, s: pre_and_post_state) -> bool {
    // sets the ith bit in p's pre
    return clear_in_prestate_(i, s.prestate);
}

fn clear_in_prestate_(i: uint, s: prestate) -> bool {
    let was_set = s.get(i);
    s.set(i, tfalse);
    return was_set != tfalse;
}

fn clear_in_postcond(i: uint, s: pre_and_post) -> bool {
    // sets the ith bit in p's post
    let was_set = s.postcondition.get(i);
    s.postcondition.set(i, tfalse);
    return was_set != tfalse;
}

// Sets all the bits in a's precondition to equal the
// corresponding bit in p's precondition.
fn set_precondition(a: ts_ann, p: precond) {
    a.conditions.precondition.become(p);
}


// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
fn set_postcondition(a: ts_ann, p: postcond) {
    a.conditions.postcondition.become(p);
}


// Sets all the bits in a's prestate to equal the
// corresponding bit in p's prestate.
fn set_prestate(a: ts_ann, p: prestate) -> bool {
    a.states.prestate.become(p)
}


// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
fn set_poststate(a: ts_ann, p: poststate) -> bool {
    a.states.poststate.become(p)
}


// Set all the bits in p that are set in new
fn extend_prestate(p: prestate, newv: poststate) -> bool {
    p.union(newv)
}


// Set all the bits in p that are set in new
fn extend_poststate(p: poststate, newv: poststate) -> bool {
    p.union(newv)
}

// Sets the given bit in p to "don't care"
fn relax_prestate(i: uint, p: prestate) -> bool {
    let was_set = p.get(i);
    p.set(i, dont_care);
    return was_set != dont_care;
}

// Clears the given bit in p
fn relax_poststate(i: uint, p: poststate) -> bool {
    return relax_prestate(i, p);
}

// Clears the given bit in p
fn relax_precond(i: uint, p: precond) { relax_prestate(i, p); }

// Sets all the bits in p to "don't care"
fn clear(p: precond) { p.clear(); }

// Sets all the bits in p to true
fn set(p: precond) { p.set_all(); }

fn ann_precond(a: ts_ann) -> precond { return a.conditions.precondition; }

fn ann_prestate(a: ts_ann) -> prestate { return a.states.prestate; }

fn ann_poststate(a: ts_ann) -> poststate { return a.states.poststate; }

fn pp_clone(p: pre_and_post) -> pre_and_post {
    return {precondition: clone(p.precondition),
         postcondition: clone(p.postcondition)};
}

fn clone(p: prestate) -> prestate { p.clone() }


// returns true if a implies b
// that is, returns true except if for some bits c and d,
// c = 1 and d = either 0 or "don't know"
fn implies(a: t, b: t) -> bool {
    let tmp = b.clone();
    tmp.difference(a);
    tmp.doesntcare()
}

fn trit_str(t: trit) -> ~str {
    match t { dont_care { ~"?" } ttrue { ~"1" } tfalse { ~"0" } }
}

// FIXME (#2538): Would be nice to have unit tests for some of these
// operations, as a step towards formalizing them more rigorously.

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
