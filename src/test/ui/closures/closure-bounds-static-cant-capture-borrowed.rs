// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

fn bar<F>(blk: F) where F: FnOnce() + 'static {
}

fn foo(x: &()) {
    bar(|| {
        //[base]~^ ERROR `x` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
        //[nll]~^^ ERROR borrowed data escapes
        //[nll]~| ERROR closure may outlive
        let _ = x;
    })
}

fn main() {
}
