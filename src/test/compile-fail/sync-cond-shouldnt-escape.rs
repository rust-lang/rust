// error-pattern: reference is not valid outside of its lifetime
fn main() {
    let m = ~sync::new_mutex();
    let mut cond = none;
    do m.lock_cond |c| {
        cond = some(c);
    }   
    option::unwrap(cond).signal();
}
