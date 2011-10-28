
obj ob<shar K>(k: K) {
    fn foo(it: block(~{a: K})) { it(~{a: k}); }
}

fn x(o: ob<str>) { o.foo() {|_i|}; }

fn main() { let o = ob::<str>("hi" + "there"); x(o); }
