//! A type representing either success or failure

use cmp::Eq;
use either::Either;

/// The result type
enum Result<T, U> {
    /// Contains the successful result value
    Ok(T),
    /// Contains the error value
    Err(U)
}

/**
 * Get the value out of a successful result
 *
 * # Failure
 *
 * If the result is an error
 */
pure fn get<T: Copy, U>(res: Result<T, U>) -> T {
    match res {
      Ok(t) => t,
      Err(the_err) => unsafe {
        fail fmt!("get called on error result: %?", the_err)
      }
    }
}

/**
 * Get a reference to the value out of a successful result
 *
 * # Failure
 *
 * If the result is an error
 */
pure fn get_ref<T, U>(res: &a/Result<T, U>) -> &a/T {
    match *res {
        Ok(ref t) => t,
        Err(ref the_err) => unsafe {
            fail fmt!("get_ref called on error result: %?", the_err)
        }
    }
}

/**
 * Get the value out of an error result
 *
 * # Failure
 *
 * If the result is not an error
 */
pure fn get_err<T, U: Copy>(res: Result<T, U>) -> U {
    match res {
      Err(u) => u,
      Ok(_) => fail ~"get_err called on ok result"
    }
}

/// Returns true if the result is `ok`
pure fn is_ok<T, U>(res: Result<T, U>) -> bool {
    match res {
      Ok(_) => true,
      Err(_) => false
    }
}

/// Returns true if the result is `err`
pure fn is_err<T, U>(res: Result<T, U>) -> bool {
    !is_ok(res)
}

/**
 * Convert to the `either` type
 *
 * `ok` result variants are converted to `either::right` variants, `err`
 * result variants are converted to `either::left`.
 */
pure fn to_either<T: Copy, U: Copy>(res: Result<U, T>) -> Either<T, U> {
    match res {
      Ok(res) => either::Right(res),
      Err(fail_) => either::Left(fail_)
    }
}

/**
 * Call a function based on a previous result
 *
 * If `res` is `ok` then the value is extracted and passed to `op` whereupon
 * `op`s result is returned. if `res` is `err` then it is immediately
 * returned. This function can be used to compose the results of two
 * functions.
 *
 * Example:
 *
 *     let res = chain(read_file(file)) { |buf|
 *         ok(parse_bytes(buf))
 *     }
 */
fn chain<T, U: Copy, V: Copy>(res: Result<T, V>, op: fn(T) -> Result<U, V>)
    -> Result<U, V> {
    match res {
      Ok(t) => op(t),
      Err(e) => Err(e)
    }
}

/**
 * Call a function based on a previous result
 *
 * If `res` is `err` then the value is extracted and passed to `op`
 * whereupon `op`s result is returned. if `res` is `ok` then it is
 * immediately returned.  This function can be used to pass through a
 * successful result while handling an error.
 */
fn chain_err<T: Copy, U: Copy, V: Copy>(
    res: Result<T, V>,
    op: fn(V) -> Result<T, U>)
    -> Result<T, U> {
    match res {
      Ok(t) => Ok(t),
      Err(v) => op(v)
    }
}

/**
 * Call a function based on a previous result
 *
 * If `res` is `ok` then the value is extracted and passed to `op` whereupon
 * `op`s result is returned. if `res` is `err` then it is immediately
 * returned. This function can be used to compose the results of two
 * functions.
 *
 * Example:
 *
 *     iter(read_file(file)) { |buf|
 *         print_buf(buf)
 *     }
 */
fn iter<T, E>(res: Result<T, E>, f: fn(T)) {
    match res {
      Ok(t) => f(t),
      Err(_) => ()
    }
}

/**
 * Call a function based on a previous result
 *
 * If `res` is `err` then the value is extracted and passed to `op` whereupon
 * `op`s result is returned. if `res` is `ok` then it is immediately returned.
 * This function can be used to pass through a successful result while
 * handling an error.
 */
fn iter_err<T, E>(res: Result<T, E>, f: fn(E)) {
    match res {
      Ok(_) => (),
      Err(e) => f(e)
    }
}

/**
 * Call a function based on a previous result
 *
 * If `res` is `ok` then the value is extracted and passed to `op` whereupon
 * `op`s result is wrapped in `ok` and returned. if `res` is `err` then it is
 * immediately returned.  This function can be used to compose the results of
 * two functions.
 *
 * Example:
 *
 *     let res = map(read_file(file)) { |buf|
 *         parse_bytes(buf)
 *     }
 */
fn map<T, E: Copy, U: Copy>(res: Result<T, E>, op: fn(T) -> U)
  -> Result<U, E> {
    match res {
      Ok(t) => Ok(op(t)),
      Err(e) => Err(e)
    }
}

/**
 * Call a function based on a previous result
 *
 * If `res` is `err` then the value is extracted and passed to `op` whereupon
 * `op`s result is wrapped in an `err` and returned. if `res` is `ok` then it
 * is immediately returned.  This function can be used to pass through a
 * successful result while handling an error.
 */
fn map_err<T: Copy, E, F: Copy>(res: Result<T, E>, op: fn(E) -> F)
  -> Result<T, F> {
    match res {
      Ok(t) => Ok(t),
      Err(e) => Err(op(e))
    }
}

impl<T, E> Result<T, E> {
    fn is_ok() -> bool { is_ok(self) }

    fn is_err() -> bool { is_err(self) }

    fn iter(f: fn(T)) {
        match self {
          Ok(t) => f(t),
          Err(_) => ()
        }
    }

    fn iter_err(f: fn(E)) {
        match self {
          Ok(_) => (),
          Err(e) => f(e)
        }
    }
}

impl<T: Copy, E> Result<T, E> {
    fn get() -> T { get(self) }

    fn map_err<F:Copy>(op: fn(E) -> F) -> Result<T,F> {
        match self {
          Ok(t) => Ok(t),
          Err(e) => Err(op(e))
        }
    }
}

impl<T, E: Copy> Result<T, E> {
    fn get_err() -> E { get_err(self) }

    fn map<U:Copy>(op: fn(T) -> U) -> Result<U,E> {
        match self {
          Ok(t) => Ok(op(t)),
          Err(e) => Err(e)
        }
    }
}

impl<T: Copy, E: Copy> Result<T, E> {
    fn chain<U:Copy>(op: fn(T) -> Result<U,E>) -> Result<U,E> {
        chain(self, op)
    }

    fn chain_err<F:Copy>(op: fn(E) -> Result<T,F>) -> Result<T,F> {
        chain_err(self, op)
    }
}

/**
 * Maps each element in the vector `ts` using the operation `op`.  Should an
 * error occur, no further mappings are performed and the error is returned.
 * Should no error occur, a vector containing the result of each map is
 * returned.
 *
 * Here is an example which increments every integer in a vector,
 * checking for overflow:
 *
 *     fn inc_conditionally(x: uint) -> result<uint,str> {
 *         if x == uint::max_value { return err("overflow"); }
 *         else { return ok(x+1u); }
 *     }
 *     map(~[1u, 2u, 3u], inc_conditionally).chain {|incd|
 *         assert incd == ~[2u, 3u, 4u];
 *     }
 */
fn map_vec<T,U:Copy,V:Copy>(
    ts: &[T], op: fn((&T)) -> Result<V,U>) -> Result<~[V],U> {

    let mut vs: ~[V] = ~[];
    vec::reserve(vs, vec::len(ts));
    for vec::each_ref(ts) |t| {
        match op(t) {
          Ok(v) => vec::push(vs, v),
          Err(u) => return Err(u)
        }
    }
    return Ok(move vs);
}

fn map_opt<T,U:Copy,V:Copy>(
    o_t: Option<T>, op: fn(T) -> Result<V,U>) -> Result<Option<V>,U> {

    match o_t {
      None => Ok(None),
      Some(t) => match op(t) {
        Ok(v) => Ok(Some(v)),
        Err(e) => Err(e)
      }
    }
}

/**
 * Same as map, but it operates over two parallel vectors.
 *
 * A precondition is used here to ensure that the vectors are the same
 * length.  While we do not often use preconditions in the standard
 * library, a precondition is used here because result::t is generally
 * used in 'careful' code contexts where it is both appropriate and easy
 * to accommodate an error like the vectors being of different lengths.
 */
fn map_vec2<S,T,U:Copy,V:Copy>(ss: &[S], ts: &[T],
                               op: fn(S,T) -> Result<V,U>) -> Result<~[V],U> {

    assert vec::same_length(ss, ts);
    let n = vec::len(ts);
    let mut vs = ~[];
    vec::reserve(vs, n);
    let mut i = 0u;
    while i < n {
        match op(ss[i],ts[i]) {
          Ok(v) => vec::push(vs, v),
          Err(u) => return Err(u)
        }
        i += 1u;
    }
    return Ok(move vs);
}

/**
 * Applies op to the pairwise elements from `ss` and `ts`, aborting on
 * error.  This could be implemented using `map2()` but it is more efficient
 * on its own as no result vector is built.
 */
fn iter_vec2<S,T,U:Copy>(ss: &[S], ts: &[T],
                         op: fn(S,T) -> Result<(),U>) -> Result<(),U> {

    assert vec::same_length(ss, ts);
    let n = vec::len(ts);
    let mut i = 0u;
    while i < n {
        match op(ss[i],ts[i]) {
          Ok(()) => (),
          Err(u) => return Err(u)
        }
        i += 1u;
    }
    return Ok(());
}

/// Unwraps a result, assuming it is an `ok(T)`
fn unwrap<T, U>(+res: Result<T, U>) -> T {
    match move res {
      Ok(move t) => move t,
      Err(_) => fail ~"unwrap called on an err result"
    }
}

/// Unwraps a result, assuming it is an `err(U)`
fn unwrap_err<T, U>(+res: Result<T, U>) -> U {
    match move res {
      Err(move u) => move u,
      Ok(_) => fail ~"unwrap called on an ok result"
    }
}

impl<T:Eq,U:Eq> Result<T,U> : Eq {
    pure fn eq(&&other: Result<T,U>) -> bool {
        match self {
            Ok(e0a) => {
                match other {
                    Ok(e0b) => e0a == e0b,
                    _ => false
                }
            }
            Err(e0a) => {
                match other {
                    Err(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&&other: Result<T,U>) -> bool { !self.eq(other) }
}

#[cfg(test)]
#[allow(non_implicitly_copyable_typarams)]
mod tests {
    fn op1() -> result::Result<int, ~str> { result::Ok(666) }

    fn op2(&&i: int) -> result::Result<uint, ~str> {
        result::Ok(i as uint + 1u)
    }

    fn op3() -> result::Result<int, ~str> { result::Err(~"sadface") }

    #[test]
    fn chain_success() {
        assert get(chain(op1(), op2)) == 667u;
    }

    #[test]
    fn chain_failure() {
        assert get_err(chain(op3(), op2)) == ~"sadface";
    }

    #[test]
    fn test_impl_iter() {
        let mut valid = false;
        Ok::<~str, ~str>(~"a").iter(|_x| valid = true);
        assert valid;

        Err::<~str, ~str>(~"b").iter(|_x| valid = false);
        assert valid;
    }

    #[test]
    fn test_impl_iter_err() {
        let mut valid = true;
        Ok::<~str, ~str>(~"a").iter_err(|_x| valid = false);
        assert valid;

        valid = false;
        Err::<~str, ~str>(~"b").iter_err(|_x| valid = true);
        assert valid;
    }

    #[test]
    fn test_impl_map() {
        assert Ok::<~str, ~str>(~"a").map(|_x| ~"b") == Ok(~"b");
        assert Err::<~str, ~str>(~"a").map(|_x| ~"b") == Err(~"a");
    }

    #[test]
    fn test_impl_map_err() {
        assert Ok::<~str, ~str>(~"a").map_err(|_x| ~"b") == Ok(~"a");
        assert Err::<~str, ~str>(~"a").map_err(|_x| ~"b") == Err(~"b");
    }
}
