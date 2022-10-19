// check-fail
// known-bug
// edition: 2021

// We really should accept this, but we need implied bounds between the regions
// in a generator interior.

pub trait FutureIterator {
    type Future<'s, 'cx>: Send
    where
        's: 'cx;
}

fn call<I: FutureIterator>() -> impl Send {
    async { // a generator checked for autotrait impl `Send`
        //~^ lifetime bound not satisfied
        let x = None::<I::Future<'_, '_>>; // a type referencing GAT
        async {}.await; // a yield point
    }
}

fn call2<'a, 'b, I: FutureIterator>() -> impl Send {
    async { // a generator checked for autotrait impl `Send`
        //~^ lifetime bound not satisfied
        let x = None::<I::Future<'a, 'b>>; // a type referencing GAT
        //~^ lifetime may not live long enough
        async {}.await; // a yield point
    }
}

fn call3<'a: 'b, 'b, I: FutureIterator>() -> impl Send {
    async { // a generator checked for autotrait impl `Send`
        //~^ lifetime bound not satisfied
        let x = None::<I::Future<'a, 'b>>; // a type referencing GAT
        async {}.await; // a yield point
    }
}

fn main() {}
