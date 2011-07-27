

type adder =
    obj {
        fn add() ;
    };

obj leaf_adder(x: int) {
    fn add() { log "leaf"; log x; }
}

obj delegate_adder(a: adder) {
    fn add() { a.add(); }
}

fn main() {
    let x = delegate_adder(delegate_adder(delegate_adder(leaf_adder(10))));
    x.add();
}