//@ run-rustfix
// https://github.com/rust-lang/rust/issues/95616

fn buggy_const<const N: usize>(_a: &Option<[u8; N]>, _f: &str) -> &str { //~ERROR [E0106]
    return "";
}

fn main() {
    buggy_const(&Some([69,69,69,69,0]), "test");
}
