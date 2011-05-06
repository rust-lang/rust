use std;
import std.Vec;
import std.BitV;

fn test_0_elements() {
  auto act;
  auto exp;

  act = BitV.create(0u, false);
  exp = Vec.init_elt[uint](0u, 0u);
  // FIXME: why can't I write vec[uint]()?
  assert (BitV.eq_vec(act, exp));
}

fn test_1_element() {
  auto act;

  act = BitV.create(1u, false);
  assert (BitV.eq_vec(act, vec(0u)));

  act = BitV.create(1u, true);
  assert (BitV.eq_vec(act, vec(1u)));
}

fn test_10_elements() {
  auto act;

  // all 0
  act = BitV.create(10u, false);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // all 1
  act = BitV.create(10u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u)));

  // mixed
  act = BitV.create(10u, false);
  BitV.set(act, 0u, true);
  BitV.set(act, 1u, true);
  BitV.set(act, 2u, true);
  BitV.set(act, 3u, true);
  BitV.set(act, 4u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u)));

  // mixed
  act = BitV.create(10u, false);
  BitV.set(act, 5u, true);
  BitV.set(act, 6u, true);
  BitV.set(act, 7u, true);
  BitV.set(act, 8u, true);
  BitV.set(act, 9u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u)));

  // mixed
  act = BitV.create(10u, false);
  BitV.set(act, 0u, true);
  BitV.set(act, 3u, true);
  BitV.set(act, 6u, true);
  BitV.set(act, 9u, true);
  assert (BitV.eq_vec(act, vec(1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 1u)));
}

fn test_31_elements() {
  auto act;

  // all 0
  act = BitV.create(31u, false);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // all 1
  act = BitV.create(31u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u)));

  // mixed
  act = BitV.create(31u, false);
  BitV.set(act, 0u, true);
  BitV.set(act, 1u, true);
  BitV.set(act, 2u, true);
  BitV.set(act, 3u, true);
  BitV.set(act, 4u, true);
  BitV.set(act, 5u, true);
  BitV.set(act, 6u, true);
  BitV.set(act, 7u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // mixed
  act = BitV.create(31u, false);
  BitV.set(act, 16u, true);
  BitV.set(act, 17u, true);
  BitV.set(act, 18u, true);
  BitV.set(act, 19u, true);
  BitV.set(act, 20u, true);
  BitV.set(act, 21u, true);
  BitV.set(act, 22u, true);
  BitV.set(act, 23u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // mixed
  act = BitV.create(31u, false);
  BitV.set(act, 24u, true);
  BitV.set(act, 25u, true);
  BitV.set(act, 26u, true);
  BitV.set(act, 27u, true);
  BitV.set(act, 28u, true);
  BitV.set(act, 29u, true);
  BitV.set(act, 30u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u)));

  // mixed
  act = BitV.create(31u, false);
  BitV.set(act, 3u, true);
  BitV.set(act, 17u, true);
  BitV.set(act, 30u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 1u)));
}

fn test_32_elements() {
  auto act;

  // all 0
  act = BitV.create(32u, false);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // all 1
  act = BitV.create(32u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u)));

  // mixed
  act = BitV.create(32u, false);
  BitV.set(act, 0u, true);
  BitV.set(act, 1u, true);
  BitV.set(act, 2u, true);
  BitV.set(act, 3u, true);
  BitV.set(act, 4u, true);
  BitV.set(act, 5u, true);
  BitV.set(act, 6u, true);
  BitV.set(act, 7u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // mixed
  act = BitV.create(32u, false);
  BitV.set(act, 16u, true);
  BitV.set(act, 17u, true);
  BitV.set(act, 18u, true);
  BitV.set(act, 19u, true);
  BitV.set(act, 20u, true);
  BitV.set(act, 21u, true);
  BitV.set(act, 22u, true);
  BitV.set(act, 23u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));

  // mixed
  act = BitV.create(32u, false);
  BitV.set(act, 24u, true);
  BitV.set(act, 25u, true);
  BitV.set(act, 26u, true);
  BitV.set(act, 27u, true);
  BitV.set(act, 28u, true);
  BitV.set(act, 29u, true);
  BitV.set(act, 30u, true);
  BitV.set(act, 31u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u)));

  // mixed
  act = BitV.create(32u, false);
  BitV.set(act, 3u, true);
  BitV.set(act, 17u, true);
  BitV.set(act, 30u, true);
  BitV.set(act, 31u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u)));
}

fn test_33_elements() {
  auto act;

  // all 0
  act = BitV.create(33u, false);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u)));

  // all 1
  act = BitV.create(33u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              1u)));

  // mixed
  act = BitV.create(33u, false);
  BitV.set(act, 0u, true);
  BitV.set(act, 1u, true);
  BitV.set(act, 2u, true);
  BitV.set(act, 3u, true);
  BitV.set(act, 4u, true);
  BitV.set(act, 5u, true);
  BitV.set(act, 6u, true);
  BitV.set(act, 7u, true);
  assert (BitV.eq_vec(act, vec(1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u)));

  // mixed
  act = BitV.create(33u, false);
  BitV.set(act, 16u, true);
  BitV.set(act, 17u, true);
  BitV.set(act, 18u, true);
  BitV.set(act, 19u, true);
  BitV.set(act, 20u, true);
  BitV.set(act, 21u, true);
  BitV.set(act, 22u, true);
  BitV.set(act, 23u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u)));

  // mixed
  act = BitV.create(33u, false);
  BitV.set(act, 24u, true);
  BitV.set(act, 25u, true);
  BitV.set(act, 26u, true);
  BitV.set(act, 27u, true);
  BitV.set(act, 28u, true);
  BitV.set(act, 29u, true);
  BitV.set(act, 30u, true);
  BitV.set(act, 31u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                              0u)));

  // mixed
  act = BitV.create(33u, false);
  BitV.set(act, 3u, true);
  BitV.set(act, 17u, true);
  BitV.set(act, 30u, true);
  BitV.set(act, 31u, true);
  BitV.set(act, 32u, true);
  assert (BitV.eq_vec(act, vec(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
                              0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                              1u)));
}

fn main() {
  test_0_elements();
  test_1_element();
  test_10_elements();
  test_31_elements();
  test_32_elements();
  test_33_elements();
}
