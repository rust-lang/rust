use std;
import std::ivec;

fn main(args: vec[str]) {
    let vs: [str] = ~["hi", "there", "this", "is", "a", "vec"];
    let vvs: [[str]] = ~[ivec::from_vec(args), vs];
    for vs: [str]  in vvs { for s: str  in vs { log s; } }
}