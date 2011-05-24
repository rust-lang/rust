// -*- rust -*-

// error-pattern: unterminated block comment

/*
 * This is an un-balanced /* multi-line comment.
 */

fn main() {
  log "hello, world.";
}
