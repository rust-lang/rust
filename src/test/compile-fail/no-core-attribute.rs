// error-pattern: whatever
#[no_core];

fn main() {
    log(debug, core::int::max_value);
}