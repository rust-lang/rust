// This checks that function pointer signatures containing &mut T types
// work in a constant context: see issue #114994.
//
// check-pass

const fn use_const_fn(_f: fn(&mut String)) {
    ()
}

const fn get_some_fn() -> fn(&mut String) {
    String::clear
}

const fn some_const_fn() {
    let _f: fn(&mut String) = String::clear;
}

fn main() {}
