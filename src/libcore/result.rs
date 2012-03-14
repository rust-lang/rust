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
fn get<T: copy, U>(res: result<T, U>) -> T {
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
fn get_err<T, U: copy>(res: result<T, U>) -> U {
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
