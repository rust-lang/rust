// Ensures that putting resources inside structual types keeps
// working.

type closable = @mut bool;

resource close_res(i: closable) { *i = false; }

enum option<T> { none, some(T), }

fn sink(res: option<close_res>) { }

fn main() {
    let c = @mut true;
    sink(none);
    sink(some(close_res(c)));
    assert (!*c);
}
