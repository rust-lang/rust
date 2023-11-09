//@ check-pass

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
}
