//@ edition: 2024
//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders
//@ check-fail

trait FutureIterator: 'static {
    type Future<'s, 'cx>: Future + Send + 'cx;
}

trait IterCaller: 'static {
    type Future2<'cx>: Future + 'cx;

    fn call_2() {}
}

struct UseIter<FI1, FI2> {
    fi_1: FI1,
    fi_2: FI2,
}

impl<FI1, FI2: 'static> IterCaller for UseIter<FI1, FI2>
//~^ ERROR not all trait items implemented
where
    FI1: FutureIterator + 'static + Send,
    for<'s, 'cx> FI1::Future<'s, 'cx>: Send,
{
    fn call_2<'s, 'cx>() -> Self::Future2<'cx>
    //~^ ERROR lifetime parameters or bounds on associated function `call_2` do not match
    where
        's: 'cx,
    {
    }
}

fn main() {}
