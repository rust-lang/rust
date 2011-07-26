

obj ob[K](K k) {
    iter foo() -> @rec(K a) { put @rec(a=k); }
}

fn x(&ob[str] o) { for each (@rec(str a) i in o.foo()) { } }

fn main() { auto o = ob[str]("hi" + "there"); x(o); }