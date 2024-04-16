//@ known-bug: #121126
fn main() {
    let _n = 1i64 >> [64][4_294_967_295];
}
