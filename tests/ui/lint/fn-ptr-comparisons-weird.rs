//@ check-pass

#[derive(PartialEq, Eq)]
struct A {
    f: fn(),
    //~^ WARN function pointer comparisons
}

fn main() {
    let f: fn() = main;
    let g: fn() = main;

    let _ = f > g;
    //~^ WARN function pointer comparisons
    let _ = f >= g;
    //~^ WARN function pointer comparisons
    let _ = f <= g;
    //~^ WARN function pointer comparisons
    let _ = f < g;
    //~^ WARN function pointer comparisons
    let _ = assert_eq!(g, g);
    //~^ WARN function pointer comparisons
    let _ = assert_ne!(g, g);
    //~^ WARN function pointer comparisons
}
