/*
Module: result

A type representing either success or failure
*/

/* Section: Types */

/*
Tag: t

The result type
*/
tag t<T, U> {
    /*
    Variant: ok

    Contains the result value
    */
    ok(T);
    /*
    Variant: error

    Contains the error value
    */
    error(U);
}

/* Section: Operations */

/*
Function: get

Get the value out of a successful result

Failure:

If the result is an error
*/
fn get<T, U>(res: t<T, U>) -> T {
    alt res {
      ok(t) { t }
      error(_) {
        fail "get called on error result";
      }
    }
}

/*
Function: get

Get the value out of an error result

Failure:

If the result is not an error
*/
fn get_error<T, U>(res: t<T, U>) -> U {
    alt res {
      error(u) { u }
      ok(_) {
        fail "get_error called on ok result";
      }
    }
}

/*
Function: success

Returns true if the result is <ok>
*/
fn success<T, U>(res: t<T, U>) -> bool {
    alt res {
      ok(_) { true }
      error(_) { false }
    }
}

/*
Function: failure

Returns true if the result is <error>
*/
fn failure<T, U>(res: t<T, U>) -> bool {
    !success(res)
}