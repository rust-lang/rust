import core::*;

// -*- rust -*-
use std;
import str;
import uint;
import str::bytes;

#[test]
fn test_from_str() {
    assert (uint::from_str("0") == 0u);
    assert (uint::from_str("3") == 3u);
    assert (uint::from_str("10") == 10u);
    assert (uint::from_str("123456789") == 123456789u);
    assert (uint::from_str("00100") == 100u);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_from_str_fail_1() {
    uint::from_str(" ");
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_from_str_fail_2() {
    uint::from_str("x");
}

#[test]
fn test_parse_buf() {
    assert (uint::parse_buf(bytes("123"), 10u) == 123u);
    assert (uint::parse_buf(bytes("1001"), 2u) == 9u);
    assert (uint::parse_buf(bytes("123"), 8u) == 83u);
    assert (uint::parse_buf(bytes("123"), 16u) == 291u);
    assert (uint::parse_buf(bytes("ffff"), 16u) == 65535u);
    assert (uint::parse_buf(bytes("z"), 36u) == 35u);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_parse_buf_fail_1() {
    uint::parse_buf(bytes("Z"), 10u);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_parse_buf_fail_2() {
    uint::parse_buf(bytes("_"), 2u);
}

#[test]
fn test_next_power_of_two() {
    assert (uint::next_power_of_two(0u) == 0u);
    assert (uint::next_power_of_two(1u) == 1u);
    assert (uint::next_power_of_two(2u) == 2u);
    assert (uint::next_power_of_two(3u) == 4u);
    assert (uint::next_power_of_two(4u) == 4u);
    assert (uint::next_power_of_two(5u) == 8u);
    assert (uint::next_power_of_two(6u) == 8u);
    assert (uint::next_power_of_two(7u) == 8u);
    assert (uint::next_power_of_two(8u) == 8u);
    assert (uint::next_power_of_two(9u) == 16u);
    assert (uint::next_power_of_two(10u) == 16u);
    assert (uint::next_power_of_two(11u) == 16u);
    assert (uint::next_power_of_two(12u) == 16u);
    assert (uint::next_power_of_two(13u) == 16u);
    assert (uint::next_power_of_two(14u) == 16u);
    assert (uint::next_power_of_two(15u) == 16u);
    assert (uint::next_power_of_two(16u) == 16u);
    assert (uint::next_power_of_two(17u) == 32u);
    assert (uint::next_power_of_two(18u) == 32u);
    assert (uint::next_power_of_two(19u) == 32u);
    assert (uint::next_power_of_two(20u) == 32u);
    assert (uint::next_power_of_two(21u) == 32u);
    assert (uint::next_power_of_two(22u) == 32u);
    assert (uint::next_power_of_two(23u) == 32u);
    assert (uint::next_power_of_two(24u) == 32u);
    assert (uint::next_power_of_two(25u) == 32u);
    assert (uint::next_power_of_two(26u) == 32u);
    assert (uint::next_power_of_two(27u) == 32u);
    assert (uint::next_power_of_two(28u) == 32u);
    assert (uint::next_power_of_two(29u) == 32u);
    assert (uint::next_power_of_two(30u) == 32u);
    assert (uint::next_power_of_two(31u) == 32u);
    assert (uint::next_power_of_two(32u) == 32u);
    assert (uint::next_power_of_two(33u) == 64u);
    assert (uint::next_power_of_two(34u) == 64u);
    assert (uint::next_power_of_two(35u) == 64u);
    assert (uint::next_power_of_two(36u) == 64u);
    assert (uint::next_power_of_two(37u) == 64u);
    assert (uint::next_power_of_two(38u) == 64u);
    assert (uint::next_power_of_two(39u) == 64u);
}

#[test]
fn test_overflows() {
   assert (uint::max_value > 0u);
   assert (uint::min_value <= 0u);
   assert (uint::min_value + uint::max_value + 1u == 0u);
}

#[test]
fn test_div() {
    assert(uint::div_floor(3u, 4u) == 0u);
    assert(uint::div_ceil(3u, 4u)  == 1u);
    assert(uint::div_round(3u, 4u) == 1u);
}