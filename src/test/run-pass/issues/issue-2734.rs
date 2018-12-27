// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

trait hax {
    fn dummy(&self) { }
}
impl<A> hax for A { }

fn perform_hax<T: 'static>(x: Box<T>) -> Box<hax+'static> {
    box x as Box<hax+'static>
}

fn deadcode() {
    perform_hax(box "deadcode".to_string());
}

pub fn main() {
    let _ = perform_hax(box 42);
}
