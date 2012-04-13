// xfail-test
/*
  tjc: currently this results in a memory leak after a call to
  span_fatal in typeck. I think it's the same issue as #2272, because
  if I make type_needs_unwind_cleanup always return true, the test passes.
  FIXME: Un-xfail this when #2272 is fixed.
 */
class cat implements int { //! ERROR can only implement interface types
  let meows: uint;
  new(in_x : uint) { self.meows = in_x; }
}

fn main() {
  let nyan = cat(0u);
}