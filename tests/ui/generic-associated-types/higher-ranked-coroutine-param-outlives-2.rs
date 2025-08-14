//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

pub trait FutureIterator {
    type Future<'s, 'cx>: Send
    where
        's: 'cx;
}

fn call<I: FutureIterator>() -> impl Send {
    async { // a coroutine checked for autotrait impl `Send`
        let x = None::<I::Future<'_, '_>>; // a type referencing GAT
        async {}.await; // a yield point
    }
}

fn call2<'a: 'b, 'b, I: FutureIterator>() -> impl Send {
    async { // a coroutine checked for autotrait impl `Send`
        let x = None::<I::Future<'a, 'b>>; // a type referencing GAT
        async {}.await; // a yield point
    }
}

fn main() {}
