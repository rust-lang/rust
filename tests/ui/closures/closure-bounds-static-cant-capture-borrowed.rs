fn bar<F>(blk: F) where F: FnOnce() + 'static {
}

fn foo(x: &()) {
    bar(|| {
        //~^ ERROR borrowed data escapes
        //~| ERROR closure may outlive
        let _ = x;
    })
}

fn main() {
}
