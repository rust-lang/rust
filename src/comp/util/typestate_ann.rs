import front.ast.ident;
import std._vec;
import std.bitv;

/* 
   This says: this expression requires the idents in <pre> to be initialized,
   and given the precondition, it guarantees that the idents in <post> are
   initialized.
 */
type precond  = bitv.t; /* 1 means "this variable must be initialized"
                           0 means "don't care about this variable" */
type postcond = bitv.t; /* 1 means "this variable is initialized"
                           0 means "don't know about this variable */

type prestate = bitv.t; /* 1 means "this variable is definitely initialized"
                           0 means "don't know whether this variable is
                           initialized" */
type poststate = bitv.t; /* 1 means "this variable is definitely initialized"
                            0 means "don't know whether this variable is
                            initialized" */

/* named thus so as not to confuse with prestate and poststate */
type pre_and_post = rec(precond precondition, postcond postcondition);
/* FIXME: once it's implemented: */
//  : ((*.precondition).nbits == (*.postcondition).nbits);

type pre_and_post_state = rec(prestate prestate, poststate poststate);

type ts_ann = rec(pre_and_post conditions, pre_and_post_state states);

fn true_precond(uint num_vars) -> precond {
  be bitv.create(num_vars, false);
}

fn true_postcond(uint num_vars) -> postcond {
  be true_precond(num_vars);
}

fn empty_prestate(uint num_vars) -> prestate {
  be true_precond(num_vars);
}

fn empty_poststate(uint num_vars) -> poststate {
  be true_precond(num_vars);
}

fn empty_pre_post(uint num_vars) -> pre_and_post {
  ret(rec(precondition=empty_prestate(num_vars),
          postcondition=empty_poststate(num_vars)));
}

fn empty_states(uint num_vars) -> pre_and_post_state {
  ret(rec(prestate=true_precond(num_vars),
           poststate=true_postcond(num_vars)));
}

fn empty_ann(uint num_vars) -> ts_ann {
  ret(rec(conditions=empty_pre_post(num_vars),
          states=empty_states(num_vars)));
}

fn get_pre(&pre_and_post p) -> precond {
  ret p.precondition;
}

fn get_post(&pre_and_post p) -> postcond {
  ret p.postcondition;
}

fn difference(&precond p1, &precond p2) -> bool {
  be bitv.difference(p1, p2);
}

fn union(&precond p1, &precond p2) -> bool {
  be bitv.difference(p1, p2);
}

fn pps_len(&pre_and_post p) -> uint {
  // gratuitous check
  check (p.precondition.nbits == p.postcondition.nbits);
  ret p.precondition.nbits;
}

impure fn require_and_preserve(uint i, &pre_and_post p) -> () {
  // sets the ith bit in p's pre and post
  bitv.set(p.precondition, i, true);
  bitv.set(p.postcondition, i, true);
}

impure fn set_in_postcond(uint i, &pre_and_post p) -> () {
  // sets the ith bit in p's post
  bitv.set(p.postcondition, i, true);
}

// Sets all the bits in a's precondition to equal the
// corresponding bit in p's precondition.
impure fn set_precondition(&ts_ann a, &precond p) -> () {
  bitv.copy(p, a.conditions.precondition);
}

// Sets all the bits in a's postcondition to equal the
// corresponding bit in p's postcondition.
impure fn set_postcondition(&ts_ann a, &postcond p) -> () {
  bitv.copy(p, a.conditions.postcondition);
}

// Set all the bits in p that are set in new
impure fn extend_prestate(&prestate p, &poststate new) -> () {
  bitv.union(p, new);
}

fn ann_precond(&ts_ann a) -> precond {
  ret a.conditions.precondition;
}

fn ann_prestate(&ts_ann a) -> prestate {
  ret a.states.prestate;
}

impure fn implies(bitv.t a, bitv.t b) -> bool {
  bitv.difference(b, a);
  be bitv.is_false(b);
}
