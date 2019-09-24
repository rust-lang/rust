// run-pass

// Tests that `loop`s unconditionally-broken-from are allowed in constants.

const FOO: () = loop { break; };

fn main() {
    [FOO; { let x; loop { x = 5; break; } x }];
}
