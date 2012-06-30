fn main(args: ~[str]) {
    let vs: ~[str] = ~["hi", "there", "this", "is", "a", "vec"];
    let vvs: ~[~[str]] = ~[args, vs];
    for vvs.each |vs| { for vs.each |s| { log(debug, s); } }
}
