#[doc = "A type representing either success or failure"];

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
      err(_) {
        // FIXME: Serialize the error value
        // and include it in the fail message (maybe just note it)
        fail "get called on error result";
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

// ______________________________________________________________________
// Note:
//
// These helper functions are written in a "pre-chained" (a.k.a,
// deforested) style because I have found that, in practice, this is
// the most concise way to do things.  That means that they do not not
// terminate with a call to `ok(v)` but rather `nxt(v)`.  If you would
// like to just get the result, just pass in `ok` as `nxt`.

#[doc = "
Maps each element in the vector `ts` using the operation `op`.  Should an
error occur, no further mappings are performed and the error is returned.
Should no error occur, a vector containing the result of each map is
passed to the `nxt` function.

Here is an example which increments every integer in a vector,
checking for overflow:

    fn inc_conditionally(x: uint) -> result<uint,str> {
        if x == uint::max_value { ret err(\"overflow\"); }
        else { ret ok(x+1u); }
    }
    map([1u, 2u, 3u], inc_conditionally) {|incd|
        assert incd == [2u, 3u, 4u];
    }

Note: if you have to combine a deforested style transform with map,
you should use `ok` for the `nxt` operation, as shown here (this is an
alternate version of the previous example where the
`inc_conditionally()` routine is deforested):

    fn inc_conditionally<T>(x: uint,
                            nxt: fn(uint) -> result<T,str>) -> result<T,str> {
        if x == uint::max_value { ret err(\"overflow\"); }
        else { ret nxt(x+1u); }
    }
    map([1u, 2u, 3u], inc_conditionally(_, ok)) {|incd|
        assert incd == [2u, 3u, 4u];
    }
"]
fn map<T,U:copy,V:copy,W>(ts: [T],
                          op: fn(T) -> result<V,U>,
                          nxt: fn([V]) -> result<W,U>) -> result<W,U> {
    let mut vs: [V] = [];
    vec::reserve(vs, vec::len(ts));
    for t in ts {
        alt op(t) {
          ok(v) { vs += [v]; }
          err(u) { ret err(u); }
        }
    }
    ret nxt(vs);
}

#[doc = "Same as map, but it operates over two parallel vectors.

A precondition is used here to ensure that the vectors are the same
length.  While we do not often use preconditions in the standard
library, a precondition is used here because result::t is generally
used in 'careful' code contexts where it is both appropriate and easy
to accommodate an error like the vectors being of different lengths."]
fn map2<S,T,U:copy,V:copy,W>(ss: [S], ts: [T],
                             op: fn(S,T) -> result<V,U>,
                             nxt: fn([V]) -> result<W,U>)
    : vec::same_length(ss, ts)
    -> result<W,U> {
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
    ret nxt(vs);
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
