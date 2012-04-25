// The two functions returning `result` should by type compatibile
// even though they use different int types. This will probably
// not be true after #2187

#[no_core];

enum result<T, U> {
    ok(T),
    err(U)
}

type error = int;

#[cfg(target_arch = "x86_64")]
fn get_fd() -> result<int, error> {
    getsockopt_i64()
}

#[cfg(target_arch = "x86_64")]
fn getsockopt_i64() -> result<i64, error> {
    fail
}

#[cfg(target_arch = "x86")]
fn get_fd() -> result<int, error> {
    getsockopt_i32()
}

#[cfg(target_arch = "x86")]
fn getsockopt_i32() -> result<i32, error> {
    fail
}

fn main() { }