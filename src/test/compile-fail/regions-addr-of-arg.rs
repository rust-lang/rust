fn foo(a: int) {
    let _p: &static.int = &a; //~ ERROR mismatched types
}

fn bar(a: int) {
    let _q: &blk.int = &a;
}

fn main() {
}