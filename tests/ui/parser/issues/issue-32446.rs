fn main() {}

// This used to end up in an infinite loop trying to bump past EOF.
trait T { ... } //~ ERROR
