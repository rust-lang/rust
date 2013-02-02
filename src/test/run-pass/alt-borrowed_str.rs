// xfail-test
// xfail-fast
// -*- rust -*-
fn f1(ref_string: &str) {
    match ref_string {
        "a" => io::println("found a"),
        "b" => io::println("found b"),
        _ => io::println("not found")
    }
}

fn f2(ref_string: &str) {
    match ref_string {
        "a" => io::println("found a"),
        "b" => io::println("found b"),
        s => io::println(fmt!("not found (%s)", s))
    }
}

fn g1(ref_1: &str, ref_2: &str) {
    match (ref_1, ref_2) {
        ("a", "b") => io::println("found a,b"),
        ("b", "c") => io::println("found b,c"),
        _ => io::println("not found")
    }
}

fn g2(ref_1: &str, ref_2: &str) {
    match (ref_1, ref_2) {
        ("a", "b") => io::println("found a,b"),
        ("b", "c") => io::println("found b,c"),
        (s1, s2) => io::println(fmt!("not found (%s, %s)", s1, s2))
    }
}

pub fn main() {
    f1(@"a");
    f1(~"b");
    f1(&"c");
    f1("d");
    f2(@"a");
    f2(~"b");
    f2(&"c");
    f2("d");
    g1(@"a", @"b");
    g1(~"b", ~"c");
    g1(&"c", &"d");
    g1("d", "e");
    g2(@"a", @"b");
    g2(~"b", ~"c");
    g2(&"c", &"d");
    g2("d", "e");
}
