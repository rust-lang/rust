

fn main(args: vec[str]) {
    let vs: vec[str] = ["hi", "there", "this", "is", "a", "vec"];
    let vvs: vec[vec[str]] = [args, vs];
    for vs: vec[str]  in vvs { for s: str  in vs { log s; } }
}