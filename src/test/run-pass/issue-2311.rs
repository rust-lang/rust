iface clam<A> { }
iface foo<A> {
   fn bar<B,C:clam<A>>(c: C) -> B;
}

fn main() { }
