thread_local!(static X: u32 = const { 0 });
//~^ ERROR: use of unstable library feature 'thread_local_const_init'

fn main() {}
