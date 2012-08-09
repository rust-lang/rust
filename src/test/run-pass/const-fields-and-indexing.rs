// Not quite working on the indexing part yet.
/*
const x : [int]/4 = [1,2,3,4];
const y : &[int] = &[1,2,3,4];
const p : int = x[2];
const q : int = y[2];
*/

const s : {a: int, b: int} = {a: 10, b: 20};
const t : int = s.b;

fn main() {

//    io::println(fmt!("%?", p));
//    io::println(fmt!("%?", q));
    io::println(fmt!("%?", t));
//    assert p == 3;
//    assert q == 3;
    assert t == 20;
}