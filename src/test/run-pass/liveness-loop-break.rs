fn test() {
    let v;
    loop {
        v = 3;
        break;
    }
    #debug["%d", v];
}

fn main() {
    test();
}
