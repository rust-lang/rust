/*
Module: either

A type that represents one of two alternatives
*/


/*
Tag: t

The either type
*/
tag t<T, U> {
    /* Variant: left */
    left(T);
    /* Variant: right */
    right(U);
}

/* Section: Operations */

/*
Function: either

Applies a function based on the given either value

If `value` is left(T) then `f_left` is applied to its contents, if
`value` is right(U) then `f_right` is applied to its contents, and
the result is returned.
*/
fn either<T, U,
          V>(f_left: block(T) -> V, f_right: block(U) -> V, value: t<T, U>) ->
   V {
    alt value { left(l) { f_left(l) } right(r) { f_right(r) } }
}

/*
Function: lefts

Extracts from a vector of either all the left values.
*/
fn lefts<copy T, U>(eithers: [t<T, U>]) -> [T] {
    let result: [T] = [];
    for elt: t<T, U> in eithers {
        alt elt { left(l) { result += [l]; } _ {/* fallthrough */ } }
    }
    ret result;
}

/*
Function: rights

Extracts from a vector of either all the right values
*/
fn rights<T, copy U>(eithers: [t<T, U>]) -> [U] {
    let result: [U] = [];
    for elt: t<T, U> in eithers {
        alt elt { right(r) { result += [r]; } _ {/* fallthrough */ } }
    }
    ret result;
}

/*
Function: partition

Extracts from a vector of either all the left values and right values

Returns a structure containing a vector of left values and a vector of
right values.
*/
fn partition<copy T, copy U>(eithers: [t<T, U>])
    -> {lefts: [T], rights: [U]} {
    let lefts: [T] = [];
    let rights: [U] = [];
    for elt: t<T, U> in eithers {
        alt elt { left(l) { lefts += [l]; } right(r) { rights += [r]; } }
    }
    ret {lefts: lefts, rights: rights};
}

/*
Function: flip

Flips between left and right of a given either
*/
pure fn flip<copy T, copy U>(eith: t<T, U>) -> t<U, T> {
    alt eith {
      right(r) { left(r) }
      left(l) { right(l) }
    }
}

/*
Function: to_result

Converts either::t to a result::t, making the "right" choice
an ok result, and the "left" choice a fail
*/
pure fn to_result<copy T, copy U>(eith: t<T, U>) -> result::t<U, T> {
    alt eith {
      right(r) { result::ok(r) }
      left(l) { result::err(l) }
    }
}

/*
Function: is_left

Checks whether the given value is a left
*/
pure fn is_left<T, U>(eith: t<T, U>) -> bool {
    alt eith { left(_) { true } _ { false } }
}

/*
Function: is_left

Checks whether the given value is a right
*/
pure fn is_right<T, U>(eith: t<T, U>) -> bool {
    alt eith { right(_) { true } _ { false } }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
