// error-pattern: Unsatisfied precondition constraint
fn send<send T>(ch: _chan<T>, -data: T) {
    log(debug, ch);
    log(debug, data);
    fail;
}
type _chan<T> = int;

// Tests that "log(debug, message);" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: _chan<int>, message: int, count: int) {
    send(ch, message);
    log(debug, message);
}

fn main() { fail; }
