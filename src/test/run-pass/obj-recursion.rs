

type adder =
    obj {
        fn add() ;
    };

obj leaf_adder(int x) {
    fn add() { log "leaf"; log x; }
}

obj delegate_adder(adder a) {
    fn add() { a.add(); }
}

fn main() {
    auto x = delegate_adder(delegate_adder(delegate_adder(leaf_adder(10))));
    x.add();
}