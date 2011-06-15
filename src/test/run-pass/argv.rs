

fn main(vec[str] args) {
    let vec[str] vs = ["hi", "there", "this", "is", "a", "vec"];
    let vec[vec[str]] vvs = [args, vs];
    for (vec[str] vs in vvs) { for (str s in vs) { log s; } }
}