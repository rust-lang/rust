fn main(args: [str]) {
    let vs: [str] = ~["hi", "there", "this", "is", "a", "vec"];
    let vvs: [[str]] = ~[args, vs];
    for vs: [str] in vvs { for s: str in vs { log s; } }
}
