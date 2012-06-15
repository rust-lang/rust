fn main() {
    let x = ~1;
    let lam_move = fn@(move x) { };
    lam_move();
}
