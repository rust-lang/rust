// xfail-test -- flat_map_to_vec currently disable

fn repeat(&&x: uint) -> [uint] { [x, x] }

fn incd_if_even(&&x: uint) -> option<uint> {
    if (x % 2u) == 0u {some(x + 1u)} else {none}
}

fn main() {
    assert [1u, 3u].flat_map_to_vec(repeat) == [1u, 1u, 3u, 3u];
    assert [].flat_map_to_vec(repeat) == [];
    assert none.flat_map_to_vec(repeat) == [];
    assert some(1u).flat_map_to_vec(repeat) == [1u, 1u];
    assert some(2u).flat_map_to_vec(repeat) == [2u, 2u];

    assert [1u, 2u, 5u].flat_map_to_vec(incd_if_even) == [3u];
    assert [].flat_map_to_vec(incd_if_even) == [];
    assert none.flat_map_to_vec(incd_if_even) == [];
    assert some(1u).flat_map_to_vec(incd_if_even) == [];
    assert some(2u).flat_map_to_vec(incd_if_even) == [3u];
}