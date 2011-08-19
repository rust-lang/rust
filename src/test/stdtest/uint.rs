

// -*- rust -*-
use std;
import std::uint;

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
