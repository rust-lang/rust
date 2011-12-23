fn test_stack_assign() {
    let s: str = "a";
    log(debug, s);
    let t: str = "a";
    assert (s == t);
    let u: str = "b";
    assert (s != u);
}

fn test_heap_lit() { "a big string"; }

fn test_heap_assign() {
    let s: str = "a big ol' string";
    let t: str = "a big ol' string";
    assert (s == t);
    let u: str = "a bad ol' string";
    assert (s != u);
}

fn test_heap_log() { let s = "a big ol' string"; log(debug, s); }

fn test_stack_add() {
    assert ("a" + "b" == "ab");
    let s: str = "a";
    assert (s + s == "aa");
    assert ("" + "" == "");
}

fn test_stack_heap_add() { assert ("a" + "bracadabra" == "abracadabra"); }

fn test_heap_add() {
    assert ("this should" + " totally work" == "this should totally work");
}

fn test_append() {
    let s = "";
    s += "a";
    assert (s == "a");

    let s = "a";
    s += "b";
    log(debug, s);
    assert (s == "ab");

    let s = "c";
    s += "offee";
    assert (s == "coffee");

    s += "&tea";
    assert (s == "coffee&tea");
}

fn main() {
    test_stack_assign();
    test_heap_lit();
    test_heap_assign();
    test_heap_log();
    test_stack_add();
    test_stack_heap_add();
    test_heap_add();
    test_append();
}
