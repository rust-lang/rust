// xfail-stage0

use std;
import std::either::*;
import std::ivec::len;

fn test_either_left() {
  auto val = left(10);
  fn f_left(&int x) -> bool { x == 10 }
  fn f_right(&uint x) -> bool { false }
  assert (either(f_left, f_right, val));
}

fn test_either_right() {
  auto val = right(10u);
  fn f_left(&int x) -> bool { false }
  fn f_right(&uint x) -> bool { x == 10u }
  assert (either(f_left, f_right, val));
}

fn test_lefts() {
  auto input = ~[left(10),
                 right(11),
                 left(12),
                 right(13),
                 left(14)];
  auto result = lefts(input);
  assert (result == ~[10, 12, 14]);
}

fn test_lefts_none() {
  let (t[int, int])[] input = ~[right(10),
                                right(10)];
  auto result = lefts(input);
  assert (len(result) == 0u);
}

fn test_lefts_empty() {
  let (t[int, int])[] input = ~[];
  auto result = lefts(input);
  assert (len(result) == 0u);
}

fn test_rights() {
  auto input = ~[left(10),
                 right(11),
                 left(12),
                 right(13),
                 left(14)];
  auto result = rights(input);
  assert (result == ~[11, 13]);
}

fn test_rights_none() {
  let (t[int, int])[] input = ~[left(10),
                                left(10)];
  auto result = rights(input);
  assert (len(result) == 0u);
}

fn test_rights_empty() {
    let (t[int, int])[] input = ~[];
    auto result = rights(input);
    assert (len(result) == 0u);
}

fn test_partition() {
  auto input = ~[left(10),
                 right(11),
                 left(12),
                 right(13),
                 left(14)];
  auto result = partition(input);
  assert (result._0.(0) == 10);
  assert (result._0.(1) == 12);
  assert (result._0.(2) == 14);
  assert (result._1.(0) == 11);
  assert (result._1.(1) == 13);
}

fn test_partition_no_lefts() {
  let (t[int, int])[] input = ~[right(10),
                                right(11)];
  auto result = partition(input);
  assert (len(result._0) == 0u);
  assert (len(result._1) == 2u);
}

fn test_partition_no_rights() {
  let (t[int, int])[] input = ~[left(10),
                                left(11)];
  auto result = partition(input);
  assert (len(result._0) == 2u);
  assert (len(result._1) == 0u);
}

fn test_partition_empty() {
  let (t[int, int])[] input = ~[];
  auto result = partition(input);
  assert (len(result._0) == 0u);
  assert (len(result._1) == 0u);
}

fn main() {
  test_either_left();
  test_either_right();
  test_lefts();
  test_lefts_none();
  test_lefts_empty();
  test_rights();
  test_rights_none();
  test_rights_empty();
  test_partition();
  test_partition_no_lefts();
  test_partition_no_rights();
  test_partition_empty();
}
