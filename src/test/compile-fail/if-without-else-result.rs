// error-pattern:`if` without `else` can not produce a result

fn main() {
    let a = if true { true };
    log_full(core::debug, a);
}