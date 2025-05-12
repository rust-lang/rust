fn infer_in_arg() {
    let x = |b: Vec<_>| {}; //~ ERROR E0282
}

fn empty_pattern() {
    let x = |_| {}; //~ ERROR type annotations needed
}

fn infer_ty() {
    let x = |k: _| {}; //~ ERROR type annotations needed
}

fn ambig_return() {
    let x = || -> Vec<_> { Vec::new() }; //~ ERROR type annotations needed
}

fn main() {}
