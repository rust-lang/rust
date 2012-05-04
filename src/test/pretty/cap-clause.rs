// pp-exact

fn main() {
    let x = 1;
    let y = 2;
    let z = 3;
    let l1 = fn@(w: int, copy x) -> int { w + x + y };
    let l2 = fn@(w: int, copy x, move y) -> int { w + x + y };
    let l3 = fn@(w: int, move z) -> int { w + z };

    let x = 1;
    let y = 2;
    let z = 3;
    let s1 = fn~(copy x) -> int { x + y };
    let s2 = fn~(copy x, move y) -> int { x + y };
    let s3 = fn~(move z) -> int { z };
}
