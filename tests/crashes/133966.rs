//@ known-bug: #133966
pub struct Data([[&'static str]; 5_i32]);
const _: &'static Data = unsafe { &*(&[] as *const Data) };
