type point = { x: int, y: int };

fn a() {
    let mut p = ~[mut 1];

    // Create an immutable pointer into p's contents:
    let _q: &int = &p[0]; //! NOTE loan of mutable vec content granted here

    p[0] = 5; //! ERROR assigning to mutable vec content prohibited due to outstanding loan
}

fn borrow(_x: &[int], _f: fn()) {}

fn b() {
    // here we alias the mutable vector into an imm slice and try to
    // modify the original:

    let mut p = ~[mut 1];

    borrow(p) {|| //! NOTE loan of mutable vec content granted here
        p[0] = 5; //! ERROR assigning to mutable vec content prohibited due to outstanding loan
    }
}

fn c() {
    // Legal because the scope of the borrow does not include the
    // modification:
    let mut p = ~[mut 1];
    borrow(p, {||});
    p[0] = 5;
}

fn main() {
}

