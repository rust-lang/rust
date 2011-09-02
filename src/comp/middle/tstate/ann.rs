
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
type pre_and_post = @{precondition: precond, postcondition: postcond};


/* FIXME: once it's implemented: */

//  : ((*.precondition).nbits == (*.postcondition).nbits);
type pre_and_post_state = {prestate: prestate, poststate: poststate};

type ts_ann = @{conditions: pre_and_post, states: pre_and_post_state};

fn true_precond(num_vars: uint) -> precond { be create_tritv(num_vars); }

fn true_postcond(num_vars: uint) -> postcond { be true_precond(num_vars); }

fn empty_prestate(num_vars: uint) -> prestate { be true_precond(num_vars); }

fn empty_poststate(num_vars: uint) -> poststate { be true_precond(num_vars); }

fn false_postcond(num_vars: uint) -> postcond {
    let rslt = create_tritv(num_vars);
    tritv_set_all(rslt);
    ret rslt;
}

fn empty_pre_post(num_vars: uint) -> pre_and_post {
    ret @{precondition: empty_prestate(num_vars),
          postcondition: empty_poststate(num_vars)};
}

fn empty_states(num_vars: uint) -> pre_and_post_state {
    ret {prestate: true_precond(num_vars),
         poststate: true_postcond(num_vars)};
}

fn empty_ann(num_vars: uint) -> ts_ann {
    ret @{conditions: empty_pre_post(num_vars),
          states: empty_states(num_vars)};
}

fn get_pre(p: &pre_and_post) -> precond { ret p.precondition; }

fn get_post(p: &pre_and_post) -> postcond { ret p.postcondition; }

fn difference(p1: &precond, p2: &precond) -> bool {
    ret tritv_difference(p1, p2);
}

fn union(p1: &precond, p2: &precond) -> bool { ret tritv_union(p1, p2); }

fn intersect(p1: &precond, p2: &precond) -> bool {
    ret tritv_intersect(p1, p2);
}

fn pps_len(p: &pre_and_post) -> uint {
    // gratuitous check

    assert (p.precondition.nbits == p.postcondition.nbits);
    ret p.precondition.nbits;
}

fn require(i: uint, p: &pre_and_post) {
    // sets the ith bit in p's pre
    tritv_set(i, p.precondition, ttrue);
}

fn require_and_preserve(i: uint, p: &pre_and_post) {
    // sets the ith bit in p's pre and post
    tritv_set(i, p.precondition, ttrue);
    tritv_set(i, p.postcondition, ttrue);
}

fn set_in_postcond(i: uint, p: &pre_and_post) -> bool {
    // sets the ith bit in p's post
    ret set_in_postcond_(i, p.postcondition);
}

fn set_in_postcond_(i: uint, p: &postcond) -> bool {
    let was_set = tritv_get(p, i);
    tritv_set(i, p, ttrue);
    ret was_set != ttrue;
}

fn set_in_poststate(i: uint, s: &pre_and_post_state) -> bool {
    // sets the ith bit in p's post
    ret set_in_poststate_(i, s.poststate);
}

fn set_in_poststate_(i: uint, p: &poststate) -> bool {
    let was_set = tritv_get(p, i);
    tritv_set(i, p, ttrue);
    ret was_set != ttrue;

}

fn clear_in_poststate(i: uint, s: &pre_and_post_state) -> bool {
    // sets the ith bit in p's post
    ret clear_in_poststate_(i, s.poststate);
}

fn clear_in_poststate_(i: uint, s: &poststate) -> bool {
    let was_set = tritv_get(s, i);
    tritv_set(i, s, tfalse);
    ret was_set != tfalse;
}

fn clear_in_prestate(i: uint, s: &pre_and_post_state) -> bool {
    // sets the ith bit in p's pre
    ret clear_in_prestate_(i, s.prestate);
}

fn clear_in_prestate_(i: uint, s: &prestate) -> bool {
    let was_set = tritv_get(s, i);
    tritv_set(i, s, tfalse);
    ret was_set != tfalse;
}

fn clear_in_postcond(i: uint, s: &pre_and_post) -> bool {
    // sets the ith bit in p's post
    let was_set = tritv_get(s.postcondition, i);
    tritv_set(i, s.postcondition, tfalse);
    ret was_set != tfalse;
}

// Sets all the bits in a's precondition to equal the
// corresponding bit in p's precondition.
fn set_precondition(a: ts_ann, p: &precond) {
    tritv_copy(a.conditions.precondition, p);
}


// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
fn set_postcondition(a: ts_ann, p: &postcond) {
    tritv_copy(a.conditions.postcondition, p);
}


// Sets all the bits in a's prestate to equal the
// corresponding bit in p's prestate.
fn set_prestate(a: ts_ann, p: &prestate) -> bool {
    ret tritv_copy(a.states.prestate, p);
}


// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
fn set_poststate(a: ts_ann, p: &poststate) -> bool {
    ret tritv_copy(a.states.poststate, p);
}


// Set all the bits in p that are set in new
fn extend_prestate(p: &prestate, new: &poststate) -> bool {
    ret tritv_union(p, new);
}


// Set all the bits in p that are set in new
fn extend_poststate(p: &poststate, new: &poststate) -> bool {
    ret tritv_union(p, new);
}

// Sets the given bit in p to "don't care"
// FIXME: is this correct?
fn relax_prestate(i: uint, p: &prestate) -> bool {
    let was_set = tritv_get(p, i);
    tritv_set(i, p, dont_care);
    ret was_set != dont_care;
}

// Clears the given bit in p
fn relax_poststate(i: uint, p: &poststate) -> bool {
    ret relax_prestate(i, p);
}

// Clears the given bit in p
fn relax_precond(i: uint, p: &precond) { relax_prestate(i, p); }

// Sets all the bits in p to "don't care"
fn clear(p: &precond) { tritv_clear(p); }

// Sets all the bits in p to true
fn set(p: &precond) { tritv_set_all(p); }

fn ann_precond(a: &ts_ann) -> precond { ret a.conditions.precondition; }

fn ann_prestate(a: &ts_ann) -> prestate { ret a.states.prestate; }

fn ann_poststate(a: &ts_ann) -> poststate { ret a.states.poststate; }

fn pp_clone(p: &pre_and_post) -> pre_and_post {
    ret @{precondition: clone(p.precondition),
          postcondition: clone(p.postcondition)};
}

fn clone(p: prestate) -> prestate { ret tritv_clone(p); }


// returns true if a implies b
// that is, returns true except if for some bits c and d,
// c = 1 and d = either 0 or "don't know"
// FIXME: is this correct?
fn implies(a: t, b: t) -> bool {
    let tmp = tritv_clone(b);
    tritv_difference(tmp, a);
    ret tritv_doesntcare(tmp);
}

fn trit_str(t: trit) -> istr {
    alt t { dont_care. { ~"?" } ttrue. { ~"1" } tfalse. { ~"0" } }
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
