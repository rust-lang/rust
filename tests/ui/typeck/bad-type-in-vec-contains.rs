// The error message here still is pretty confusing.

fn main() {
    let primes = Vec::new();
    primes.contains(3);
    //~^ ERROR mismatched types
}
