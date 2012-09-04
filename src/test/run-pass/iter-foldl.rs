fn add(&&x: float, &&y: uint) -> float { x + (y as float) }

fn main() {
    assert [1u, 3u]/_.foldl(20f, add) == 24f;
    assert []/_.foldl(20f, add) == 20f;
    assert None.foldl(20f, add) == 20f;
    assert Some(1u).foldl(20f, add) == 21f;
    assert Some(2u).foldl(20f, add) == 22f;
}
