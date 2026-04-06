#![forbid(for_loops_over_fallibles)]

fn main() {
    macro_rules! x {
        () => {
            None::<i32>
        };
    }
    for _ in x! {} {} //~ ERROR for loop over an `Option`. This is more readably written as an `if let` statement [for_loops_over_fallibles]
}
