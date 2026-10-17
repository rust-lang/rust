//@ check-pass

#[derive(PartialEq, Eq)]
struct A {
    f: fn(),
    //~^ WARN function pointer comparisons
}

#[allow(unpredictable_function_pointer_comparisons)]
#[derive(PartialEq, Eq)]
struct AllowedAbove {
    f: fn(),
}

#[derive(PartialEq, Eq)]
#[allow(unpredictable_function_pointer_comparisons)]
struct AllowedBelow {
    f: fn(),
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
