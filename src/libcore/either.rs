/*
Module: either

A type that represents one of two alternatives
*/


/*
Tag: t

The either type
*/
enum t<T, U> {
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
fn lefts<T: copy, U>(eithers: [t<T, U>]) -> [T] {
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
fn rights<T, U: copy>(eithers: [t<T, U>]) -> [U] {
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
fn partition<T: copy, U: copy>(eithers: [t<T, U>])
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
pure fn flip<T: copy, U: copy>(eith: t<T, U>) -> t<U, T> {
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
pure fn to_result<T: copy, U: copy>(eith: t<T, U>) -> result::t<U, T> {
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

#[test]
fn test_either_left() {
    let val = left(10);
    fn f_left(&&x: int) -> bool { x == 10 }
    fn f_right(&&_x: uint) -> bool { false }
    assert (either(f_left, f_right, val));
}

#[test]
fn test_either_right() {
    let val = right(10u);
    fn f_left(&&_x: int) -> bool { false }
    fn f_right(&&x: uint) -> bool { x == 10u }
    assert (either(f_left, f_right, val));
}

#[test]
fn test_lefts() {
    let input = [left(10), right(11), left(12), right(13), left(14)];
    let result = lefts(input);
    assert (result == [10, 12, 14]);
}

#[test]
fn test_lefts_none() {
    let input: [t<int, int>] = [right(10), right(10)];
    let result = lefts(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_lefts_empty() {
    let input: [t<int, int>] = [];
    let result = lefts(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_rights() {
    let input = [left(10), right(11), left(12), right(13), left(14)];
    let result = rights(input);
    assert (result == [11, 13]);
}

#[test]
fn test_rights_none() {
    let input: [t<int, int>] = [left(10), left(10)];
    let result = rights(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_rights_empty() {
    let input: [t<int, int>] = [];
    let result = rights(input);
    assert (vec::len(result) == 0u);
}

#[test]
fn test_partition() {
    let input = [left(10), right(11), left(12), right(13), left(14)];
    let result = partition(input);
    assert (result.lefts[0] == 10);
    assert (result.lefts[1] == 12);
    assert (result.lefts[2] == 14);
    assert (result.rights[0] == 11);
    assert (result.rights[1] == 13);
}

#[test]
fn test_partition_no_lefts() {
    let input: [t<int, int>] = [right(10), right(11)];
    let result = partition(input);
    assert (vec::len(result.lefts) == 0u);
    assert (vec::len(result.rights) == 2u);
}

#[test]
fn test_partition_no_rights() {
    let input: [t<int, int>] = [left(10), left(11)];
    let result = partition(input);
    assert (vec::len(result.lefts) == 2u);
    assert (vec::len(result.rights) == 0u);
}

#[test]
fn test_partition_empty() {
    let input: [t<int, int>] = [];
    let result = partition(input);
    assert (vec::len(result.lefts) == 0u);
    assert (vec::len(result.rights) == 0u);
}
