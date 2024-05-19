fn main() {
    // This used to ICE during liveness check because `target_id` passed to
    // `propagate_through_expr` would be the closure and not the `loop`, which wouldn't be found in
    // `self.break_ln`. (#62480)
    'a: {
        || break 'a
        //~^ ERROR use of unreachable label `'a`
        //~| ERROR `break` inside of a closure
    }
}
