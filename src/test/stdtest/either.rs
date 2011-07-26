use std;
import std::either::*;
import std::ivec::len;

#[test]
fn test_either_left() {
  auto val = left(10);
  fn f_left(&int x) -> bool { x == 10 }
  fn f_right(&uint x) -> bool { false }
  assert (either(f_left, f_right, val));
}

#[test]
fn test_either_right() {
  auto val = right(10u);
  fn f_left(&int x) -> bool { false }
  fn f_right(&uint x) -> bool { x == 10u }
  assert (either(f_left, f_right, val));
}

#[test]
fn test_lefts() {
  auto input = ~[left(10),
                 right(11),
                 left(12),
                 right(13),
                 left(14)];
  auto result = lefts(input);
  assert (result == ~[10, 12, 14]);
}

#[test]
fn test_lefts_none() {
  let (t[int, int])[] input = ~[right(10),
                                right(10)];
  auto result = lefts(input);
  assert (len(result) == 0u);
}

#[test]
fn test_lefts_empty() {
  let (t[int, int])[] input = ~[];
  auto result = lefts(input);
  assert (len(result) == 0u);
}

#[test]
fn test_rights() {
  auto input = ~[left(10),
                 right(11),
                 left(12),
                 right(13),
                 left(14)];
  auto result = rights(input);
  assert (result == ~[11, 13]);
}

#[test]
fn test_rights_none() {
  let (t[int, int])[] input = ~[left(10),
                                left(10)];
  auto result = rights(input);
  assert (len(result) == 0u);
}

#[test]
fn test_rights_empty() {
    let (t[int, int])[] input = ~[];
    auto result = rights(input);
    assert (len(result) == 0u);
}

#[test]
fn test_partition() {
  auto input = ~[left(10),
                 right(11),
                 left(12),
                 right(13),
                 left(14)];
  auto result = partition(input);
  assert (result.lefts.(0) == 10);
  assert (result.lefts.(1) == 12);
  assert (result.lefts.(2) == 14);
  assert (result.rights.(0) == 11);
  assert (result.rights.(1) == 13);
}

#[test]
fn test_partition_no_lefts() {
  let (t[int, int])[] input = ~[right(10),
                                right(11)];
  auto result = partition(input);
  assert (len(result.lefts) == 0u);
  assert (len(result.rights) == 2u);
}

#[test]
fn test_partition_no_rights() {
  let (t[int, int])[] input = ~[left(10),
                                left(11)];
  auto result = partition(input);
  assert (len(result.lefts) == 2u);
  assert (len(result.rights) == 0u);
}

#[test]
fn test_partition_empty() {
  let (t[int, int])[] input = ~[];
  auto result = partition(input);
  assert (len(result.lefts) == 0u);
  assert (len(result.rights) == 0u);
}
