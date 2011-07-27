// -*- rust -*-

// error-pattern: non-predicate

fn main() {
    check (1 ==
               2); // should fail to typecheck, as (a == b)
                   // is not a manifest call

}