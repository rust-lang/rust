// error-pattern: Unsatisfied precondition constraint
fn send<send T>(ch: _chan<T>, -data: T) {
    log_full(core::debug, ch);
    log_full(core::debug, data);
    fail;
}
type _chan<T> = int;

// Tests that "log_full(core::debug, message);" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: _chan<int>, message: int, count: int) {
    send(ch, message);
    log_full(core::debug, message);
}

fn main() { fail; }
