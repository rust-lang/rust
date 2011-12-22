

fn main() { let x = @mutable 5; *x = 1000; log_full(core::debug, *x); }
