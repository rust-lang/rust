#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn assert_static(_t: &'static u8) {}
fn main() {
     assert_static(&FOO); //~ ERROR [E0712]
}
