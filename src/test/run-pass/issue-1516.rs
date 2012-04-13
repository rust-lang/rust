// xfail-test
fn main() {  let early_error: fn@(str) -> !  = {|msg| fail }; }

