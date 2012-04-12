fn add(&&x: float, &&y: uint) -> float { x + (y as float) }

fn main() {
    assert [1u, 3u].foldl(20f, add) == 24f;
    assert [].foldl(20f, add) == 20f;
    assert none.foldl(20f, add) == 20f;
    assert some(1u).foldl(20f, add) == 21f;
    assert some(2u).foldl(20f, add) == 22f;
}