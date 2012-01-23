/*
Module: result

A type representing either success or failure
*/

/* Section: Types */

/*
Tag: t

The result type
*/
enum t<T, U> {
    /*
    Variant: ok

    Contains the result value
    */
    ok(T),
    /*
    Variant: err

    Contains the error value
    */
    err(U)
}

/* Section: Operations */

/*
Function: get

Get the value out of a successful result

Failure:

If the result is an error
*/
fn get<T: copy, U>(res: t<T, U>) -> T {
    alt res {
      ok(t) { t }
      err(_) {
        // FIXME: Serialize the error value
        // and include it in the fail message (maybe just note it)
        fail "get called on error result";
      }
    }
}

/*
Function: get_err

Get the value out of an error result

Failure:

If the result is not an error
*/
fn get_err<T, U: copy>(res: t<T, U>) -> U {
    alt res {
      err(u) { u }
      ok(_) {
        fail "get_error called on ok result";
      }
    }
}

/*
Function: success

Returns true if the result is <ok>
*/
pure fn success<T, U>(res: t<T, U>) -> bool {
    alt res {
      ok(_) { true }
      err(_) { false }
    }
}

/*
Function: failure

Returns true if the result is <error>
*/
pure fn failure<T, U>(res: t<T, U>) -> bool {
    !success(res)
}

pure fn to_either<T: copy, U: copy>(res: t<U, T>) -> either::t<T, U> {
    alt res {
      ok(res) { either::right(res) }
      err(fail_) { either::left(fail_) }
    }
}

/*
Function: chain

Call a function based on a previous result

If `res` is <ok> then the value is extracted and passed to `op` whereupon
`op`s result is returned. if `res` is <err> then it is immediately returned.
This function can be used to compose the results of two functions.

Example:

> let res = chain(read_file(file), { |buf|
>   ok(parse_buf(buf))
> })

*/
fn chain<T, U: copy, V: copy>(res: t<T, V>, op: fn(T) -> t<U, V>)
    -> t<U, V> {
    alt res {
      ok(t) { op(t) }
      err(e) { err(e) }
    }
}

#[cfg(test)]
mod tests {
    fn op1() -> result::t<int, str> { result::ok(666) }

    fn op2(&&i: int) -> result::t<uint, str> { result::ok(i as uint + 1u) }

    fn op3() -> result::t<int, str> { result::err("sadface") }

    #[test]
    fn chain_success() {
        assert get(chain(op1(), op2)) == 667u;
    }

    #[test]
    fn chain_failure() {
        assert get_err(chain(op3(), op2)) == "sadface";
    }
}
