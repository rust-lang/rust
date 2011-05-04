// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// error-pattern: token

/*
 * This is an un-balanced /* multi-line comment.
 */

fn main() {
  log "hello, world.";
}
