// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
