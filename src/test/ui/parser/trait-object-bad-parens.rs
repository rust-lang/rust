// compile-flags: -Z parse-only -Z continue-parse-after-error

fn main() {
    let _: Box<((Copy)) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `((Copy))`
    let _: Box<(Copy + Copy) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(Copy + Copy)`
    let _: Box<(Copy +) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(Copy)`
    let _: Box<(dyn Copy) + Copy>;
    //~^ ERROR expected a path on the left-hand side of `+`, not `(dyn Copy)`
}
