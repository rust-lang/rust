
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
type pre_and_post = @rec(precond precondition, postcond postcondition);


/* FIXME: once it's implemented: */

//  : ((*.precondition).nbits == (*.postcondition).nbits);
type pre_and_post_state = rec(prestate prestate, poststate poststate);

type ts_ann = @rec(pre_and_post conditions, pre_and_post_state states);

fn true_precond(uint num_vars) -> precond {
    be create_tritv(num_vars);
}

fn true_postcond(uint num_vars) -> postcond { be true_precond(num_vars); }

fn empty_prestate(uint num_vars) -> prestate { be true_precond(num_vars); }

fn empty_poststate(uint num_vars) -> poststate { be true_precond(num_vars); }

fn false_postcond(uint num_vars) -> postcond {
    auto rslt = create_tritv(num_vars);
    tritv_set_all(rslt);
    ret rslt;
}

fn empty_pre_post(uint num_vars) -> pre_and_post {
    ret @rec(precondition=empty_prestate(num_vars),
             postcondition=empty_poststate(num_vars));
}

fn empty_states(uint num_vars) -> pre_and_post_state {
    ret rec(prestate=true_precond(num_vars),
            poststate=true_postcond(num_vars));
}

fn empty_ann(uint num_vars) -> ts_ann {
    ret @rec(conditions=empty_pre_post(num_vars),
             states=empty_states(num_vars));
}

fn get_pre(&pre_and_post p) -> precond { ret p.precondition; }

fn get_post(&pre_and_post p) -> postcond { ret p.postcondition; }

fn difference(&precond p1, &precond p2) -> bool {
    ret tritv_difference(p1, p2);
}

fn union(&precond p1, &precond p2) -> bool {
    ret tritv_union(p1, p2);
}

fn intersect(&precond p1, &precond p2) -> bool {
    ret tritv_intersect(p1, p2);
}

fn pps_len(&pre_and_post p) -> uint {
    // gratuitous check

    assert (p.precondition.nbits == p.postcondition.nbits);
    ret p.precondition.nbits;
}

fn require(uint i, &pre_and_post p) {
    // sets the ith bit in p's pre
    tritv_set(i, p.precondition, ttrue);
}

fn require_and_preserve(uint i, &pre_and_post p) {
    // sets the ith bit in p's pre and post
    tritv_set(i, p.precondition, ttrue);
    tritv_set(i, p.postcondition, ttrue);
}

fn set_in_postcond(uint i, &pre_and_post p) -> bool {
    // sets the ith bit in p's post
    auto was_set = tritv_get(p.postcondition, i);
    tritv_set(i, p.postcondition, ttrue);
    ret was_set != ttrue;
}

fn set_in_poststate(uint i, &pre_and_post_state s) -> bool {
    // sets the ith bit in p's post
    ret set_in_poststate_(i, s.poststate);
}

fn set_in_poststate_(uint i, &poststate p) -> bool {
    auto was_set = tritv_get(p, i);
    tritv_set(i, p, ttrue);
    ret was_set != ttrue;

}

fn clear_in_poststate(uint i, &pre_and_post_state s) -> bool {
    // sets the ith bit in p's post
    ret clear_in_poststate_(i, s.poststate);
}

fn clear_in_poststate_(uint i, &poststate s) -> bool {
    auto was_set = tritv_get(s, i);
    tritv_set(i, s, tfalse);
    ret was_set != tfalse;
}

fn clear_in_prestate(uint i, &pre_and_post_state s) -> bool {
    // sets the ith bit in p's pre
    ret clear_in_prestate_(i, s.prestate);
}

fn clear_in_prestate_(uint i, &prestate s) -> bool {
    auto was_set = tritv_get(s, i);
    tritv_set(i, s, tfalse);
    ret was_set != tfalse;
}

fn clear_in_postcond(uint i, &pre_and_post s) -> bool {
    // sets the ith bit in p's post
    auto was_set = tritv_get(s.postcondition, i);
    tritv_set(i, s.postcondition, tfalse);
    ret was_set != tfalse;
}

// Sets all the bits in a's precondition to equal the
// corresponding bit in p's precondition.
fn set_precondition(ts_ann a, &precond p) {
    tritv_copy(a.conditions.precondition, p);
}


// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
fn set_postcondition(ts_ann a, &postcond p) {
    tritv_copy(a.conditions.postcondition, p);
}


// Sets all the bits in a's prestate to equal the
// corresponding bit in p's prestate.
fn set_prestate(ts_ann a, &prestate p) -> bool {
    ret tritv_copy(a.states.prestate, p);
}


// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
fn set_poststate(ts_ann a, &poststate p) -> bool {
    ret tritv_copy(a.states.poststate, p);
}


// Set all the bits in p that are set in new
fn extend_prestate(&prestate p, &poststate new) -> bool {
    ret tritv_union(p, new);
}


// Set all the bits in p that are set in new
fn extend_poststate(&poststate p, &poststate new) -> bool {
    ret tritv_union(p, new);
}

// Sets the given bit in p to "don't care"
// FIXME: is this correct?
fn relax_prestate(uint i, &prestate p) -> bool {
    auto was_set = tritv_get(p, i);
    tritv_set(i, p, dont_care);
    ret was_set != dont_care;
}

// Clears the given bit in p
fn relax_poststate(uint i, &poststate p) -> bool {
    ret relax_prestate(i, p);
}

// Clears the given bit in p
fn relax_precond(uint i, &precond p) {
    relax_prestate(i, p);
}

// Sets all the bits in p to "don't care"
fn clear(&precond p) { tritv_clear(p); }

// Sets all the bits in p to true
fn set(&precond p) { tritv_set_all(p); }

fn ann_precond(&ts_ann a) -> precond { ret a.conditions.precondition; }

fn ann_prestate(&ts_ann a) -> prestate { ret a.states.prestate; }

fn ann_poststate(&ts_ann a) -> poststate { ret a.states.poststate; }

fn pp_clone(&pre_and_post p) -> pre_and_post {
    ret @rec(precondition=clone(p.precondition),
             postcondition=clone(p.postcondition));
}

fn clone(prestate p) -> prestate { ret tritv_clone(p); }


// returns true if a implies b
// that is, returns true except if for some bits c and d,
// c = 1 and d = either 0 or "don't know"
// FIXME: is this correct?
fn implies(t a, t b) -> bool {
    auto tmp = tritv_clone(b);
    tritv_difference(tmp, a);
    ret tritv_doesntcare(tmp);
}

fn trit_str(trit t) -> str {
    alt (t) {
        case (dont_care) { "?" }
        case (ttrue)     { "1" }
        case (tfalse)    { "0" }
    }
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
