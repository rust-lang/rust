fn main(args: [istr]) {
    let vs: [istr] = [~"hi", ~"there", ~"this", ~"is", ~"a", ~"vec"];
    let vvs: [[istr]] = [args, vs];
    for vs: [istr] in vvs { for s: istr in vs { log s; } }
}
