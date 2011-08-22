fn test_stack_assign() {
    let s: istr = ~"a";
    log s;
    let t: istr = ~"a";
    assert s == t;
    let u: istr = ~"b";
    assert s != u;
}

fn test_heap_lit() {
    ~"a big string";
}

fn test_heap_assign() {
    let s: istr;
    s = ~"AAAA";
}

fn test_heap_log() {
    let s = ~"a big ol' string";
    log s;
}

fn main() {
    test_stack_assign();
    test_heap_lit();
    test_heap_assign();
    test_heap_log();
}