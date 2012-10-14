// NB: transitionary, de-mode-ing.
// tjc: allowing deprecated modes due to function issue.
// can re-forbid them after snapshot
#[forbid(deprecated_pattern)];

/*!
 * A type representing values that may be computed concurrently and
 * operations for working with them.
 *
 * # Example
 *
 * ~~~
 * let delayed_fib = future::spawn {|| fib(5000) };
 * make_a_sandwich();
 * io::println(fmt!("fib(5000) = %?", delayed_fib.get()))
 * ~~~
 */

use either::Either;
use pipes::{recv, oneshot, ChanOne, PortOne, send_one, recv_one};
use cast::copy_lifetime;

#[doc = "The future type"]
pub struct Future<A> {
    /*priv*/ mut state: FutureState<A>,
}

// FIXME(#2829) -- futures should not be copyable, because they close
// over fn~'s that have pipes and so forth within!
impl<A> Future<A> : Drop {
    fn finalize() {}
}

priv enum FutureState<A> {
    Pending(fn~() -> A),
    Evaluating,
    Forced(~A)
}

/// Methods on the `future` type
impl<A:Copy> Future<A> {
    fn get() -> A {
        //! Get the value of the future

        get(&self)
    }
}

impl<A> Future<A> {
    fn get_ref(&self) -> &self/A {
        get_ref(self)
    }

    fn with<B>(blk: fn(&A) -> B) -> B {
        //! Work with the value without copying it

        with(&self, blk)
    }
}

pub fn from_value<A>(val: A) -> Future<A> {
    /*!
     * Create a future from a value
     *
     * The value is immediately available and calling `get` later will
     * not block.
     */

    Future {state: Forced(~(move val))}
}

pub fn from_port<A:Send>(port: PortOne<A>) ->
        Future<A> {
    /*!
     * Create a future from a port
     *
     * The first time that the value is requested the task will block
     * waiting for the result to be received on the port.
     */

    let port = ~mut Some(move port);
    do from_fn |move port| {
        let mut port_ = None;
        port_ <-> *port;
        let port = option::unwrap(move port_);
        match recv(move port) {
            oneshot::send(move data) => move data
        }
    }
}

pub fn from_fn<A>(f: ~fn() -> A) -> Future<A> {
    /*!
     * Create a future from a function.
     *
     * The first time that the value is requested it will be retreived by
     * calling the function.  Note that this function is a local
     * function. It is not spawned into another task.
     */

    Future {state: Pending(move f)}
}

pub fn spawn<A:Send>(blk: fn~() -> A) -> Future<A> {
    /*!
     * Create a future from a unique closure.
     *
     * The closure will be run in a new task and its result used as the
     * value of the future.
     */

    let (chan, port) = oneshot::init();

    let chan = ~mut Some(move chan);
    do task::spawn |move blk, move chan| {
        let chan = option::swap_unwrap(&mut *chan);
        send_one(move chan, blk());
    }

    return from_port(move port);
}

pub fn get_ref<A>(future: &r/Future<A>) -> &r/A {
    /*!
     * Executes the future's closure and then returns a borrowed
     * pointer to the result.  The borrowed pointer lasts as long as
     * the future.
     */

    // The unsafety here is to hide the aliases from borrowck, which
    // would otherwise be concerned that someone might reassign
    // `future.state` and cause the value of the future to be freed.
    // But *we* know that once `future.state` is `Forced()` it will
    // never become "unforced"---so we can safely return a pointer
    // into the interior of the Forced() variant which will last as
    // long as the future itself.

    match future.state {
      Forced(ref v) => { // v here has type &A, but with a shorter lifetime.
        return unsafe{ copy_lifetime(future, &**v) }; // ...extend it.
      }
      Evaluating => {
        fail ~"Recursive forcing of future!";
      }
      Pending(_) => {}
    }

    let mut state = Evaluating;
    state <-> future.state;
    match move state {
      Forced(_) | Evaluating => {
        fail ~"Logic error.";
      }
      Pending(move f) => {
        future.state = Forced(~f());
        return get_ref(future);
      }
    }
}

pub fn get<A:Copy>(future: &Future<A>) -> A {
    //! Get the value of the future

    *get_ref(future)
}

pub fn with<A,B>(future: &Future<A>, blk: fn(&A) -> B) -> B {
    //! Work with the value without copying it

    blk(get_ref(future))
}

#[allow(non_implicitly_copyable_typarams)]
pub mod test {
    #[test]
    pub fn test_from_value() {
        let f = from_value(~"snail");
        assert get(&f) == ~"snail";
    }

    #[test]
    pub fn test_from_port() {
        let (ch, po) = oneshot::init();
        send_one(move ch, ~"whale");
        let f = from_port(move po);
        assert get(&f) == ~"whale";
    }

    #[test]
    pub fn test_from_fn() {
        let f = from_fn(|| ~"brail");
        assert get(&f) == ~"brail";
    }

    #[test]
    pub fn test_interface_get() {
        let f = from_value(~"fail");
        assert f.get() == ~"fail";
    }

    #[test]
    pub fn test_with() {
        let f = from_value(~"nail");
        assert with(&f, |v| copy *v) == ~"nail";
    }

    #[test]
    pub fn test_get_ref_method() {
        let f = from_value(22);
        assert *f.get_ref() == 22;
    }

    #[test]
    pub fn test_get_ref_fn() {
        let f = from_value(22);
        assert *get_ref(&f) == 22;
    }

    #[test]
    pub fn test_interface_with() {
        let f = from_value(~"kale");
        assert f.with(|v| copy *v) == ~"kale";
    }

    #[test]
    pub fn test_spawn() {
        let f = spawn(|| ~"bale");
        assert get(&f) == ~"bale";
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    pub fn test_futurefail() {
        let f = spawn(|| fail);
        let _x: ~str = get(&f);
    }

    #[test]
    pub fn test_sendable_future() {
        let expected = ~"schlorf";
        let f = do spawn |copy expected| { copy expected };
        do task::spawn |move f, move expected| {
            let actual = get(&f);
            assert actual == expected;
        }
    }
}
