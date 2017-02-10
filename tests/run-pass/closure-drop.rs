struct Foo<'a>(&'a mut bool);

impl<'a> Drop for Foo<'a> {
    fn drop(&mut self) {
        *self.0 = true;
    }
}

fn f<T: FnOnce()>(t: T) {
    t()
}

fn main() {
    let mut ran_drop = false;
    {
        // FIXME: v is a temporary hack to force the below closure to be a FnOnce-only closure
        // (with sig fn(self)). Without it, the closure sig would be fn(&self) which requires a
        // shim to call via FnOnce::call_once, and Miri's current shim doesn't correctly call
        // destructors.
        let v = vec![1];
        let x = Foo(&mut ran_drop);
        let g = move || {
            let _ = x;
            drop(v); // Force the closure to be FnOnce-only by using a capture by-value.
        };
        f(g);
    }
    assert!(ran_drop);
}

