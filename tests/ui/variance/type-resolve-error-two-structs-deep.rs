// Make sure we don't report bivariance errors when nesting structs w/ unresolved
// fields into *other* structs.

struct Hello<'a> {
    missing: Missing<'a>,
    //~^ ERROR cannot find type `Missing` in this scope
}

struct Other<'a> {
    hello: Hello<'a>,
}

fn main() {}
