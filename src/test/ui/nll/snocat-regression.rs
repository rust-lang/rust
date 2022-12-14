// Regression test from https://github.com/rust-lang/rust/pull/98109

pub fn negotiate<S>(link: S)
where
    for<'a> &'a S: 'a,
{
    || {
        //~^ ERROR `S` does not live long enough
        //
        // FIXME(#98437). This regressed at some point and
        // probably should work.
        let _x = link;
    };
}

fn main() {}
