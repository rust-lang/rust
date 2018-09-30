// run-pass

fn main() {
    // We shouldn't promote this
    &(main as fn() == main as fn());
    // Also check nested case
    &(&(main as fn()) == &(main as fn()));
}
