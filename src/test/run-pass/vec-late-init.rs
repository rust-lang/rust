

fn main() {
    let later: [int];
    if true { later = [1]; } else { later = [2]; }
    log_full(core::debug, later[0]);
}
