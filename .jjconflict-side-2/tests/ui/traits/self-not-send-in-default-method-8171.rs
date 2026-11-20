// https://github.com/rust-lang/rust/issues/8171
//@ check-pass
#![allow(dead_code)]

/*

#8171 Self is not recognised as implementing kinds in default method implementations

*/

fn require_send<T: Send>(_: T){}

trait TragicallySelfIsNotSend: Send + Sized {
    fn x(self) {
        require_send(self);
    }
}

pub fn main(){}
