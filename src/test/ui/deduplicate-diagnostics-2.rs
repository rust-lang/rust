// revisions: duplicate deduplicate
//[deduplicate] compile-flags: -Z deduplicate-diagnostics=yes

fn main() {
    match 0.0 {
        1.0 => {} //~ ERROR floating-point types cannot be used in patterns
                  //~| WARNING this was previously accepted
                  //~| ERROR floating-point types cannot be used in patterns
                  //~| WARNING this was previously accepted
        2.0 => {} //~ ERROR floating-point types cannot be used in patterns
                  //~| WARNING this was previously accepted
                  //[duplicate]~| ERROR floating-point types cannot be used in patterns
                  //[duplicate]~| WARNING this was previously accepted
        _ => {}
    }
}
