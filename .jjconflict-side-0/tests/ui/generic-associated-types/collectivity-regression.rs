// Regression test from https://github.com/rust-lang/rust/pull/98109

pub trait Get {
    type Value<'a>
    where
        Self: 'a;
}

fn multiply_at<T>(x: T)
where
    for<'a> T: Get<Value<'a> = ()>,
{
    || {
        //~^ ERROR `T` does not live long enough
        //
        // FIXME(#98437). This regressed at some point and
        // probably should work.
        let _x = x;
    };
}

fn main() {}
