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
        let x = Foo(&mut ran_drop);
        // this closure never by val uses its captures
        // so it's basically a fn(&self)
        // the shim used to not drop the `x`
        let x = move || {
            let _val = x;
        };
        f(x);
    }
    assert!(ran_drop);
}
