#[derive(Clone)]
struct InGroup<F> {
    it: It,
    //~^ ERROR cannot find type `It` in this scope
    f: F,
}
fn dates_in_year() -> impl Clone {
    InGroup { f: |d| d }
    //~^ ERROR missing field `it` in initializer of `InGroup<_>`
}

fn main() {}
