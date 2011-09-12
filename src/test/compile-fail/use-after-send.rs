// error-pattern: Unsatisfied precondition constraint
fn send<~T>(ch: _chan<T>, -data: T) { log ch; log data; fail; }
type _chan<T> = int;

// Tests that "log message;" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: _chan<int>, message: int, count: int) {
    send(ch, message);
    log message;
}

fn main() { fail; }
