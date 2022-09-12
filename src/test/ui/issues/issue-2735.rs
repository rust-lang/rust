// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

trait hax {
    fn dummy(&self) { }
}
impl<A> hax for A { }

fn perform_hax<T: 'static>(x: Box<T>) -> Box<dyn hax+'static> {
    Box::new(x) as Box<dyn hax+'static>
}

fn deadcode() {
    perform_hax(Box::new("deadcode".to_string()));
}

pub fn main() {
    perform_hax(Box::new(42));
}
