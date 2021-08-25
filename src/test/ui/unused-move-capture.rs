// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    let _x: Box<_> = Box::new(1);
    let lam_move = || {};
    lam_move();
}
