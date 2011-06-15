

obj ob[K](K k) {
    iter foo() -> @tup(K) { put @tup(k); }
}

fn x(&ob[str] o) { for each (@tup(str) i in o.foo()) { } }

fn main() { auto o = ob[str]("hi" + "there"); x(o); }