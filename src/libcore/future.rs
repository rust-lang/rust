// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
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

import either::Either;
import pipes::recv;
import unsafe::copy_lifetime;

export Future;
export extensions;
export from_value;
export from_port;
export from_fn;
export get;
export with;
export spawn;

// for task.rs
export future_pipe;

#[doc = "The future type"]
struct Future<A> {
    /*priv*/ mut state: FutureState<A>;
}

priv enum FutureState<A> {
    Pending(fn@() -> A),
    Evaluating,
    Forced(A)
}

/// Methods on the `future` type
impl<A:copy> Future<A> {
    fn get() -> A {
        //! Get the value of the future

        get(&self)
    }
}

impl<A> Future<A> {
    fn get_ref(&self) -> &self/A {
        get_ref(self)
    }

    fn with<B>(blk: fn((&A)) -> B) -> B {
        //! Work with the value without copying it

        with(&self, blk)
    }
}

fn from_value<A>(+val: A) -> Future<A> {
    /*!
     * Create a future from a value
     *
     * The value is immediately available and calling `get` later will
     * not block.
     */

    Future {state: Forced(val)}
}

fn from_port<A:send>(+port: future_pipe::client::waiting<A>) -> Future<A> {
    /*!
     * Create a future from a port
     *
     * The first time that the value is requested the task will block
     * waiting for the result to be received on the port.
     */

    let port = ~mut Some(port);
    do from_fn |move port| {
        let mut port_ = None;
        port_ <-> *port;
        let port = option::unwrap(port_);
        match recv(port) {
          future_pipe::completed(move data) => data
        }
    }
}

fn from_fn<A>(+f: @fn() -> A) -> Future<A> {
    /*!
     * Create a future from a function.
     *
     * The first time that the value is requested it will be retreived by
     * calling the function.  Note that this function is a local
     * function. It is not spawned into another task.
     */

    Future {state: Pending(f)}
}

fn spawn<A:send>(+blk: fn~() -> A) -> Future<A> {
    /*!
     * Create a future from a unique closure.
     *
     * The closure will be run in a new task and its result used as the
     * value of the future.
     */

    from_port(pipes::spawn_service_recv(future_pipe::init, |ch| {
        future_pipe::server::completed(ch, blk());
    }))
}

fn get_ref<A>(future: &r/Future<A>) -> &r/A {
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
        return unsafe{ copy_lifetime(future, v) }; // ...extend it.
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
        future.state = Forced(f());
        return get_ref(future);
      }
    }
}

fn get<A:copy>(future: &Future<A>) -> A {
    //! Get the value of the future

    *get_ref(future)
}

fn with<A,B>(future: &Future<A>, blk: fn((&A)) -> B) -> B {
    //! Work with the value without copying it

    blk(get_ref(future))
}

proto! future_pipe (
    waiting:recv<T:send> {
        completed(T) -> !
    }
)

#[allow(non_implicitly_copyable_typarams)]
mod test {
    #[test]
    fn test_from_value() {
        let f = from_value(~"snail");
        assert get(&f) == ~"snail";
    }

    #[test]
    fn test_from_port() {
        let (po, ch) = future_pipe::init();
        future_pipe::server::completed(ch, ~"whale");
        let f = from_port(po);
        assert get(&f) == ~"whale";
    }

    #[test]
    fn test_from_fn() {
        let f = from_fn(|| ~"brail");
        assert get(&f) == ~"brail";
    }

    #[test]
    fn test_interface_get() {
        let f = from_value(~"fail");
        assert f.get() == ~"fail";
    }

    #[test]
    fn test_with() {
        let f = from_value(~"nail");
        assert with(&f, |v| copy *v) == ~"nail";
    }

    #[test]
    fn test_get_ref_method() {
        let f = from_value(22);
        assert *f.get_ref() == 22;
    }

    #[test]
    fn test_get_ref_fn() {
        let f = from_value(22);
        assert *get_ref(&f) == 22;
    }

    #[test]
    fn test_interface_with() {
        let f = from_value(~"kale");
        assert f.with(|v| copy *v) == ~"kale";
    }

    #[test]
    fn test_spawn() {
        let f = spawn(|| ~"bale");
        assert get(&f) == ~"bale";
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_futurefail() {
        let f = spawn(|| fail);
        let _x: ~str = get(&f);
    }
}