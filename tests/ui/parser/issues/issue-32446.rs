fn main() {}

// This used to end up in an infite loop trying to bump past EOF.
trait T { ... } //~ ERROR
