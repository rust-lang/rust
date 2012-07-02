fn borrow(_v: &int) {}

fn local() {
    let mut v = ~3;
    borrow(v);
}

fn local_rec() {
    let mut v = {f: ~3};
    borrow(v.f);
}

fn local_recs() {
    let mut v = {f: {g: {h: ~3}}};
    borrow(v.f.g.h);
}

fn aliased_imm() {
    let mut v = ~3;
    let _w = &v;
    borrow(v);
}

fn aliased_const() {
    let mut v = ~3;
    let _w = &const v;
    borrow(v);
}

fn aliased_mut() {
    let mut v = ~3;
    let _w = &mut v; //~ NOTE prior loan as mutable granted here
    borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
}

fn aliased_other() {
    let mut v = ~3, w = ~4;
    let _x = &mut w;
    borrow(v);
}

fn aliased_other_reassign() {
    let mut v = ~3, w = ~4;
    let mut _x = &mut w;
    _x = &mut v; //~ NOTE prior loan as mutable granted here
    borrow(v); //~ ERROR loan of mutable local variable as immutable conflicts with prior loan
}

fn main() {
}
