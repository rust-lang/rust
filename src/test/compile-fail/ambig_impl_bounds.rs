iface A { fn foo(); }
iface B { fn foo(); }

fn foo<T: A B>(t: T) {
    t.foo(); //! ERROR multiple applicable methods in scope
    //!^ NOTE candidate #1 derives from the bound `A`
    //!^^ NOTE candidate #2 derives from the bound `B`
}

fn main() {}