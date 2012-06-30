fn send<T: send>(ch: _chan<T>, -data: T) {
    log(debug, ch);
    log(debug, data);
    fail;
}

enum _chan<T> = int;

// Tests that "log(debug, message);" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: _chan<int>, message: int, _count: int) {
    send(ch, message); //~ NOTE move of variable occurred here
    log(debug, message); //~ ERROR use of moved variable: `message`
}

fn main() { fail; }
