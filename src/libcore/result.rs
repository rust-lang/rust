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
        // FIXME: have a run-fail test for this
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
pure fn success<T, U>(res: result<T, U>) -> bool {
    alt res {
      ok(_) { true }
      err(_) { false }
    }
}

#[doc = "Returns true if the result is `error`"]
pure fn failure<T, U>(res: result<T, U>) -> bool {
    !success(res)
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

impl extensions<T:copy, E:copy> for result<T,E> {
    fn get() -> T { get(self) }

    fn get_err() -> E { get_err(self) }

    fn success() -> bool { success(self) }

    fn failure() -> bool { failure(self) }

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
    map([1u, 2u, 3u], inc_conditionally).chain {|incd|
        assert incd == [2u, 3u, 4u];
    }
"]
fn map<T,U:copy,V:copy>(ts: [T], op: fn(T) -> result<V,U>) -> result<[V],U> {
    let mut vs: [V] = [];
    vec::reserve(vs, vec::len(ts));
    for vec::each(ts) {|t|
        alt op(t) {
          ok(v) { vs += [v]; }
          err(u) { ret err(u); }
        }
    }
    ret ok(vs);
}

#[doc = "Same as map, but it operates over two parallel vectors.

A precondition is used here to ensure that the vectors are the same
length.  While we do not often use preconditions in the standard
library, a precondition is used here because result::t is generally
used in 'careful' code contexts where it is both appropriate and easy
to accommodate an error like the vectors being of different lengths."]
fn map2<S,T,U:copy,V:copy>(ss: [S], ts: [T], op: fn(S,T) -> result<V,U>)
    : vec::same_length(ss, ts) -> result<[V],U> {

    let n = vec::len(ts);
    let mut vs = [];
    vec::reserve(vs, n);
    let mut i = 0u;
    while i < n {
        alt op(ss[i],ts[i]) {
          ok(v) { vs += [v]; }
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
fn iter2<S,T,U:copy>(ss: [S], ts: [T],
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
}
