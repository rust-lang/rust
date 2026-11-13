// Check that we don't blindly emit a diagnostic claiming that "`main` has an invalid return type"
// if we encounter a type that doesn't implement `std::process::Termination` and is not actually
// the return type of the program entry `main`.

fn receive(_: impl std::process::Termination) {}

struct Something;

fn main() {
    receive(Something); //~ ERROR the trait bound `Something: Termination` is not satisfied
}
