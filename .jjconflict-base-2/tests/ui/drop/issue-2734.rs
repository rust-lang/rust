//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


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
    let _ = perform_hax(Box::new(42));
}
