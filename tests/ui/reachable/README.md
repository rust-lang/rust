A variety of tests around reachability. These tests in general check
two things:

- that we get unreachable code warnings in reasonable locations;
- that we permit coercions **into** `!` from expressions which
  diverge, where an expression "diverges" if it must execute some
  subexpression of type `!`, or it has type `!` itself.
