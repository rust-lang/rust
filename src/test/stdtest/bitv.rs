
use std;
import std::vec;
import std::bitv;

#[test]
fn test_0_elements() {
    let act;
    let exp;
    act = bitv::create(0u, false);
    exp = vec::init_elt[uint](0u, 0u);
    // FIXME: why can't I write vec[uint]()?

    assert (bitv::eq_ivec(act, exp));
}

#[test]
fn test_1_element() {
    let act;
    act = bitv::create(1u, false);
    assert (bitv::eq_ivec(act, ~[0u]));
    act = bitv::create(1u, true);
    assert (bitv::eq_ivec(act, ~[1u]));
}

#[test]
fn test_10_elements() {
    let act;
    // all 0

    act = bitv::create(10u, false);
    assert (bitv::eq_ivec(act, ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u]));
    // all 1

    act = bitv::create(10u, true);
    assert (bitv::eq_ivec(act, ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(10u, false);
    bitv::set(act, 0u, true);
    bitv::set(act, 1u, true);
    bitv::set(act, 2u, true);
    bitv::set(act, 3u, true);
    bitv::set(act, 4u, true);
    assert (bitv::eq_ivec(act, ~[1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(10u, false);
    bitv::set(act, 5u, true);
    bitv::set(act, 6u, true);
    bitv::set(act, 7u, true);
    bitv::set(act, 8u, true);
    bitv::set(act, 9u, true);
    assert (bitv::eq_ivec(act, ~[0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(10u, false);
    bitv::set(act, 0u, true);
    bitv::set(act, 3u, true);
    bitv::set(act, 6u, true);
    bitv::set(act, 9u, true);
    assert (bitv::eq_ivec(act, ~[1u, 0u, 0u, 1u, 0u, 0u, 1u, 0u, 0u, 1u]));
}

#[test]
fn test_31_elements() {
    let act;
    // all 0

    act = bitv::create(31u, false);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u]));
    // all 1

    act = bitv::create(31u, true);
    assert (bitv::eq_ivec(act,
                         ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(31u, false);
    bitv::set(act, 0u, true);
    bitv::set(act, 1u, true);
    bitv::set(act, 2u, true);
    bitv::set(act, 3u, true);
    bitv::set(act, 4u, true);
    bitv::set(act, 5u, true);
    bitv::set(act, 6u, true);
    bitv::set(act, 7u, true);
    assert (bitv::eq_ivec(act,
                         ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(31u, false);
    bitv::set(act, 16u, true);
    bitv::set(act, 17u, true);
    bitv::set(act, 18u, true);
    bitv::set(act, 19u, true);
    bitv::set(act, 20u, true);
    bitv::set(act, 21u, true);
    bitv::set(act, 22u, true);
    bitv::set(act, 23u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(31u, false);
    bitv::set(act, 24u, true);
    bitv::set(act, 25u, true);
    bitv::set(act, 26u, true);
    bitv::set(act, 27u, true);
    bitv::set(act, 28u, true);
    bitv::set(act, 29u, true);
    bitv::set(act, 30u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(31u, false);
    bitv::set(act, 3u, true);
    bitv::set(act, 17u, true);
    bitv::set(act, 30u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 1u]));
}

#[test]
fn test_32_elements() {
    let act;
    // all 0

    act = bitv::create(32u, false);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u]));
    // all 1

    act = bitv::create(32u, true);
    assert (bitv::eq_ivec(act,
                         ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(32u, false);
    bitv::set(act, 0u, true);
    bitv::set(act, 1u, true);
    bitv::set(act, 2u, true);
    bitv::set(act, 3u, true);
    bitv::set(act, 4u, true);
    bitv::set(act, 5u, true);
    bitv::set(act, 6u, true);
    bitv::set(act, 7u, true);
    assert (bitv::eq_ivec(act,
                         ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(32u, false);
    bitv::set(act, 16u, true);
    bitv::set(act, 17u, true);
    bitv::set(act, 18u, true);
    bitv::set(act, 19u, true);
    bitv::set(act, 20u, true);
    bitv::set(act, 21u, true);
    bitv::set(act, 22u, true);
    bitv::set(act, 23u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(32u, false);
    bitv::set(act, 24u, true);
    bitv::set(act, 25u, true);
    bitv::set(act, 26u, true);
    bitv::set(act, 27u, true);
    bitv::set(act, 28u, true);
    bitv::set(act, 29u, true);
    bitv::set(act, 30u, true);
    bitv::set(act, 31u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(32u, false);
    bitv::set(act, 3u, true);
    bitv::set(act, 17u, true);
    bitv::set(act, 30u, true);
    bitv::set(act, 31u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 1u, 1u]));
}

#[test]
fn test_33_elements() {
    let act;
    // all 0

    act = bitv::create(33u, false);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u]));
    // all 1

    act = bitv::create(33u, true);
    assert (bitv::eq_ivec(act,
                         ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u, 1u]));
    // mixed

    act = bitv::create(33u, false);
    bitv::set(act, 0u, true);
    bitv::set(act, 1u, true);
    bitv::set(act, 2u, true);
    bitv::set(act, 3u, true);
    bitv::set(act, 4u, true);
    bitv::set(act, 5u, true);
    bitv::set(act, 6u, true);
    bitv::set(act, 7u, true);
    assert (bitv::eq_ivec(act,
                         ~[1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(33u, false);
    bitv::set(act, 16u, true);
    bitv::set(act, 17u, true);
    bitv::set(act, 18u, true);
    bitv::set(act, 19u, true);
    bitv::set(act, 20u, true);
    bitv::set(act, 21u, true);
    bitv::set(act, 22u, true);
    bitv::set(act, 23u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u]));
    // mixed

    act = bitv::create(33u, false);
    bitv::set(act, 24u, true);
    bitv::set(act, 25u, true);
    bitv::set(act, 26u, true);
    bitv::set(act, 27u, true);
    bitv::set(act, 28u, true);
    bitv::set(act, 29u, true);
    bitv::set(act, 30u, true);
    bitv::set(act, 31u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u,
                           1u, 1u, 1u, 1u, 1u, 1u, 0u]));
    // mixed

    act = bitv::create(33u, false);
    bitv::set(act, 3u, true);
    bitv::set(act, 17u, true);
    bitv::set(act, 30u, true);
    bitv::set(act, 31u, true);
    bitv::set(act, 32u, true);
    assert (bitv::eq_ivec(act,
                         ~[0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                           0u, 0u, 0u, 0u, 1u, 1u, 1u]));
}

