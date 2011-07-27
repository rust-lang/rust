// xfail-stage0
fn main() { let x: vec[int] = []; for i: int  in x { fail "moop"; } }