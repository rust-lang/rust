fn bar<F>(blk: F) where F: FnOnce() + 'static {
}

fn foo(x: &()) {
    bar(|| {
        //~^ ERROR explicit lifetime required in the type of `x` [E0621]
        let _ = x;
    })
}

fn main() {
}
