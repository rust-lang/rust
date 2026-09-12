//@ known-bug: #154779
struct Data([[&'static str]; 1]);
const _: &'static Data = &*(&[] as *const Data) ;
fn main() {}
