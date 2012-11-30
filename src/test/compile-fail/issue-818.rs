mod ctr {

    pub enum ctr { priv mkCtr(int) }

    pub fn new(i: int) -> ctr { mkCtr(i) }
    pub fn inc(c: ctr) -> ctr { mkCtr(*c + 1) }
}


fn main() {
    let c = ctr::new(42);
    let c2 = ctr::inc(c);
    assert *c2 == 5; //~ ERROR can only dereference enums with a single, public variant
}