// Ensures that putting resources inside structual types keeps
// working.

type closable = @mutable bool;

resource close_res(i: closable) {
    *i = false;
}

tag option[T] { none; some(T); }

fn sink(res: option[close_res]) {}

fn main() {
    let c = @mutable true;
    sink(none);
    sink(some(close_res(c)));
    assert !*c;
}
