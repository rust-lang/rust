// pretty-expanded FIXME #23616

macro_rules! quote_tokens { () => (()) }

pub fn main() {
    quote_tokens!();
}
