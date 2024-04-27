//@ run-pass
#![allow(unused_braces)]
#![allow(non_camel_case_types)]

#[derive(Copy, Clone, Debug)]
enum chan { chan_t, }

impl PartialEq for chan {
    fn eq(&self, other: &chan) -> bool {
        ((*self) as usize) == ((*other) as usize)
    }
    fn ne(&self, other: &chan) -> bool { !(*self).eq(other) }
}

fn wrapper3(i: chan) {
    assert_eq!(i, chan::chan_t);
}

pub fn main() {
    let wrapped = {||wrapper3(chan::chan_t)};
    wrapped();
}
