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

fn empty_pre_post(uint num_vars) -> pre_and_post {
  ret(rec(precondition=true_precond(num_vars),
           postcondition=true_postcond(num_vars)));
}

fn empty_states(uint num_vars) -> pre_and_post_state {
  ret(rec(prestate=true_precond(num_vars),
           poststate=true_postcond(num_vars)));
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

fn implies(bitv.t a, bitv.t b) -> bool {
  bitv.difference(b, a);
  ret (bitv.equal(b, bitv.create(b.nbits, false)));
}
