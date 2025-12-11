//@ aux-crate: glob_vs_expanded=glob-vs-expanded.rs

fn main() {
    glob_vs_expanded::mac!(); //~ ERROR `mac` is ambiguous
                              //~| WARN this was previously accepted
}
