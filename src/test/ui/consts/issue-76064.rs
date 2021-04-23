// Note: non-`&str` panic arguments gained a separate error in PR #80734
// which is why this doesn't match the issue
struct Bug([u8; panic!("panic")]); //~ ERROR panicking in constants is unstable

fn main() {}
