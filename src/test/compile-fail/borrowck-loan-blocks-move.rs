fn take(-_v: ~int) {
}

fn box_imm() {
    let v = ~3;
    let _w = &v; //~ NOTE loan of immutable local variable granted here
    take(v); //~ ERROR moving out of immutable local variable prohibited due to outstanding loan
}

fn main() {
}
