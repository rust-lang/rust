//@compile-flags: -Zmiri-disable-isolation

fn main() {
    assert_eq!(num_cpus::get(), 1);
}
