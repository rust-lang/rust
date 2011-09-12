use std;
import std::either::*;
import std::vec::len;

#[test]
fn test_either_left() {
    let val = left(10);
    fn f_left(x: int) -> bool { x == 10 }
    fn f_right(_x: uint) -> bool { false }
    assert (either(f_left, f_right, val));
}

#[test]
fn test_either_right() {
    let val = right(10u);
    fn f_left(_x: int) -> bool { false }
    fn f_right(x: uint) -> bool { x == 10u }
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
    assert (len(result) == 0u);
}

#[test]
fn test_lefts_empty() {
    let input: [t<int, int>] = [];
    let result = lefts(input);
    assert (len(result) == 0u);
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
    assert (len(result) == 0u);
}

#[test]
fn test_rights_empty() {
    let input: [t<int, int>] = [];
    let result = rights(input);
    assert (len(result) == 0u);
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
    assert (len(result.lefts) == 0u);
    assert (len(result.rights) == 2u);
}

#[test]
fn test_partition_no_rights() {
    let input: [t<int, int>] = [left(10), left(11)];
    let result = partition(input);
    assert (len(result.lefts) == 2u);
    assert (len(result.rights) == 0u);
}

#[test]
fn test_partition_empty() {
    let input: [t<int, int>] = [];
    let result = partition(input);
    assert (len(result.lefts) == 0u);
    assert (len(result.rights) == 0u);
}
