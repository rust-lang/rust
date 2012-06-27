#[doc = "A type representing either success or failure"];

import either::either;

#[doc = "The result type"]
enum result<T, U> {
    #[doc = "Contains the successful result value"]
    ok(T),
    #[doc = "Contains the error value"]
    err(U)
}

#[doc = "
Get the value out of a successful result

# Failure

If the result is an error
"]
pure fn get<T: copy, U>(res: result<T, U>) -> T {
    alt res {
      ok(t) { t }
      err(the_err) {
        unchecked{ fail #fmt("get called on error result: %?", the_err); }
      }
    }
}

#[doc = "
Get the value out of an error result

# Failure

If the result is not an error
"]
pure fn get_err<T, U: copy>(res: result<T, U>) -> U {
    alt res {
      err(u) { u }
      ok(_) {
        fail "get_error called on ok result";
      }
    }
}

#[doc = "Returns true if the result is `ok`"]
pure fn is_ok<T, U>(res: result<T, U>) -> bool {
    alt res {
      ok(_) { true }
      err(_) { false }
    }
}

#[doc = "Returns true if the result is `err`"]
pure fn is_err<T, U>(res: result<T, U>) -> bool {
    !is_ok(res)
}

#[doc = "
Convert to the `either` type

`ok` result variants are converted to `either::right` variants, `err`
result variants are converted to `either::left`.
"]
pure fn to_either<T: copy, U: copy>(res: result<U, T>) -> either<T, U> {
    alt res {
      ok(res) { either::right(res) }
      err(fail_) { either::left(fail_) }
    }
}

#[doc = "
Call a function based on a previous result

If `res` is `ok` then the value is extracted and passed to `op` whereupon
`op`s result is returned. if `res` is `err` then it is immediately returned.
This function can be used to compose the results of two functions.

Example:

    let res = chain(read_file(file)) { |buf|
        ok(parse_buf(buf))
    }
"]
fn chain<T, U: copy, V: copy>(res: result<T, V>, op: fn(T) -> result<U, V>)
    -> result<U, V> {
    alt res {
      ok(t) { op(t) }
      err(e) { err(e) }
    }
}

#[doc = "
Call a function based on a previous result

If `res` is `err` then the value is extracted and passed to `op`
whereupon `op`s result is returned. if `res` is `ok` then it is
immediately returned.  This function can be used to pass through a
successful result while handling an error.
"]
fn chain_err<T: copy, U: copy, V: copy>(
    res: result<T, V>,
    op: fn(V) -> result<T, U>)
    -> result<T, U> {
    alt res {
      ok(t) { ok(t) }
      err(v) { op(v) }
    }
}

#[doc = "
Call a function based on a previous result

If `res` is `ok` then the value is extracted and passed to `op` whereupon
`op`s result is returned. if `res` is `err` then it is immediately returned.
This function can be used to compose the results of two functions.

Example:

    iter(read_file(file)) { |buf|
        print_buf(buf)
    }
"]
fn iter<T, E>(res: result<T, E>, f: fn(T)) {
    alt res {
      ok(t) { f(t) }
      err(_) { }
    }
}

#[doc = "
Call a function based on a previous result

If `res` is `err` then the value is extracted and passed to `op` whereupon
`op`s result is returned. if `res` is `ok` then it is immediately returned.
This function can be used to pass through a successful result while handling
an error.
"]
fn iter_err<T, E>(res: result<T, E>, f: fn(E)) {
    alt res {
      ok(_) { }
      err(e) { f(e) }
    }
}

#[doc = "
Call a function based on a previous result

If `res` is `ok` then the value is extracted and passed to `op` whereupon
`op`s result is wrapped in `ok` and returned. if `res` is `err` then it is
immediately returned.  This function can be used to compose the results of two
functions.

Example:

    let res = map(read_file(file)) { |buf|
        parse_buf(buf)
    }
"]
fn map<T, E: copy, U: copy>(res: result<T, E>, op: fn(T) -> U)
  -> result<U, E> {
    alt res {
      ok(t) { ok(op(t)) }
      err(e) { err(e) }
    }
}

#[doc = "
Call a function based on a previous result

If `res` is `err` then the value is extracted and passed to `op` whereupon
`op`s result is wrapped in an `err` and returned. if `res` is `ok` then it is
immediately returned.  This function can be used to pass through a successful
result while handling an error.
"]
fn map_err<T: copy, E, F: copy>(res: result<T, E>, op: fn(E) -> F)
  -> result<T, F> {
    alt res {
      ok(t) { ok(t) }
      err(e) { err(op(e)) }
    }
}

impl extensions<T, E> for result<T, E> {
    fn is_ok() -> bool { is_ok(self) }

    fn is_err() -> bool { is_err(self) }

    fn iter(f: fn(T)) {
        alt self {
          ok(t) { f(t) }
          err(_) { }
        }
    }

    fn iter_err(f: fn(E)) {
        alt self {
          ok(_) { }
          err(e) { f(e) }
        }
    }
}

impl extensions<T:copy, E> for result<T, E> {
    fn get() -> T { get(self) }

    fn map_err<F:copy>(op: fn(E) -> F) -> result<T,F> {
        alt self {
          ok(t) { ok(t) }
          err(e) { err(op(e)) }
        }
    }
}

impl extensions<T, E:copy> for result<T, E> {
    fn get_err() -> E { get_err(self) }

    fn map<U:copy>(op: fn(T) -> U) -> result<U,E> {
        alt self {
          ok(t) { ok(op(t)) }
          err(e) { err(e) }
        }
    }
}

impl extensions<T:copy, E:copy> for result<T,E> {
    fn chain<U:copy>(op: fn(T) -> result<U,E>) -> result<U,E> {
        chain(self, op)
    }

    fn chain_err<F:copy>(op: fn(E) -> result<T,F>) -> result<T,F> {
        chain_err(self, op)
    }
}

#[doc = "
Maps each element in the vector `ts` using the operation `op`.  Should an
error occur, no further mappings are performed and the error is returned.
Should no error occur, a vector containing the result of each map is
returned.

Here is an example which increments every integer in a vector,
checking for overflow:

    fn inc_conditionally(x: uint) -> result<uint,str> {
        if x == uint::max_value { ret err(\"overflow\"); }
        else { ret ok(x+1u); }
    }
    map([1u, 2u, 3u]/~, inc_conditionally).chain {|incd|
        assert incd == [2u, 3u, 4u]/~;
    }
"]
fn map_vec<T,U:copy,V:copy>(
    ts: [T]/~, op: fn(T) -> result<V,U>) -> result<[V]/~,U> {

    let mut vs: [V]/~ = []/~;
    vec::reserve(vs, vec::len(ts));
    for vec::each(ts) {|t|
        alt op(t) {
          ok(v) { vec::push(vs, v); }
          err(u) { ret err(u); }
        }
    }
    ret ok(vs);
}

fn map_opt<T,U:copy,V:copy>(
    o_t: option<T>, op: fn(T) -> result<V,U>) -> result<option<V>,U> {

    alt o_t {
      none { ok(none) }
      some(t) {
        alt op(t) {
          ok(v) { ok(some(v)) }
          err(e) { err(e) }
        }
      }
    }
}

#[doc = "Same as map, but it operates over two parallel vectors.

A precondition is used here to ensure that the vectors are the same
length.  While we do not often use preconditions in the standard
library, a precondition is used here because result::t is generally
used in 'careful' code contexts where it is both appropriate and easy
to accommodate an error like the vectors being of different lengths."]
fn map_vec2<S,T,U:copy,V:copy>(ss: [S]/~, ts: [T]/~,
                               op: fn(S,T) -> result<V,U>)
    : vec::same_length(ss, ts) -> result<[V]/~,U> {

    let n = vec::len(ts);
    let mut vs = []/~;
    vec::reserve(vs, n);
    let mut i = 0u;
    while i < n {
        alt op(ss[i],ts[i]) {
          ok(v) { vec::push(vs, v); }
          err(u) { ret err(u); }
        }
        i += 1u;
    }
    ret ok(vs);
}

#[doc = "
Applies op to the pairwise elements from `ss` and `ts`, aborting on
error.  This could be implemented using `map2()` but it is more efficient
on its own as no result vector is built.
"]
fn iter_vec2<S,T,U:copy>(ss: [S]/~, ts: [T]/~,
                         op: fn(S,T) -> result<(),U>)
    : vec::same_length(ss, ts)
    -> result<(),U> {

    let n = vec::len(ts);
    let mut i = 0u;
    while i < n {
        alt op(ss[i],ts[i]) {
          ok(()) { }
          err(u) { ret err(u); }
        }
        i += 1u;
    }
    ret ok(());
}

#[doc="
Unwraps a result, assuming it is an `ok(T)`
"]
fn unwrap<T, U>(-res: result<T, U>) -> T {
    unsafe {
        let addr = alt res {
          ok(x) { ptr::addr_of(x) }
          err(_) { fail "error result" }
        };
        let liberated_value = unsafe::reinterpret_cast(*addr);
        unsafe::forget(res);
        ret liberated_value;
    }
}

#[cfg(test)]
mod tests {
    fn op1() -> result::result<int, str> { result::ok(666) }

    fn op2(&&i: int) -> result::result<uint, str> {
        result::ok(i as uint + 1u)
    }

    fn op3() -> result::result<int, str> { result::err("sadface") }

    #[test]
    fn chain_success() {
        assert get(chain(op1(), op2)) == 667u;
    }

    #[test]
    fn chain_failure() {
        assert get_err(chain(op3(), op2)) == "sadface";
    }

    #[test]
    fn test_impl_iter() {
        let mut valid = false;
        ok::<str, str>("a").iter { |_x| valid = true; };
        assert valid;

        err::<str, str>("b").iter { |_x| valid = false; };
        assert valid;
    }

    #[test]
    fn test_impl_iter_err() {
        let mut valid = true;
        ok::<str, str>("a").iter_err { |_x| valid = false; };
        assert valid;

        valid = false;
        err::<str, str>("b").iter_err { |_x| valid = true; };
        assert valid;
    }

    #[test]
    fn test_impl_map() {
        assert ok::<str, str>("a").map { |_x| "b" } == ok("b");
        assert err::<str, str>("a").map { |_x| "b" } == err("a");
    }

    #[test]
    fn test_impl_map_err() {
        assert ok::<str, str>("a").map_err { |_x| "b" } == ok("a");
        assert err::<str, str>("a").map_err { |_x| "b" } == err("b");
    }
}
