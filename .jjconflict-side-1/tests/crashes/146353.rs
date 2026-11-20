//@ known-bug: #146353
const BIG_CHAIN: u8 = ();
trait NeverSend = !Send;
fn main() {}
