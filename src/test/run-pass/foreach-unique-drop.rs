
obj ob<K>(k: K) {
    iter foo() -> ~{a: K} { put ~{a: k}; }
}

fn x(o: ob<str>) { for each i: ~{a: str} in o.foo() { } }

fn main() { let o = ob::<str>("hi" + "there"); x(o); }
