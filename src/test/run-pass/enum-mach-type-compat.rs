// The two functions returning `result` should by type compatibile
// even though they use different int types. This will probably
// not be true after #2187

#[no_core];

enum result<T, U> {
    ok(T),
    err(U)
}

type error = int;

fn get_fd() -> result<int, error> {
    getsockopt_i64()
}

fn getsockopt_i64() -> result<i64, error> {
    fail
}

fn main() { }