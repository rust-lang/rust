// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

#![feature(box_syntax)]

trait hax {
    fn dummy(&self) { }
}
impl<A> hax for A { }

fn perform_hax<T: 'static>(x: Box<T>) -> Box<dyn hax+'static> {
    box x as Box<dyn hax+'static>
}

fn deadcode() {
    perform_hax(box "deadcode".to_string());
}

pub fn main() {
    perform_hax(box 42);
}
