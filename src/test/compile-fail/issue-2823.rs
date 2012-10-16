struct C {
    x: int,
    drop {
        #error("dropping: %?", self.x);
    }
}

fn main() {
    let c = C{ x: 2};
    let d = copy c; //~ ERROR copying a noncopyable value
    #error("%?", d.x);
}