// run-fail
// error-pattern:panicked at 'Box<Any>'

#![feature(box_syntax)]

fn main() {
    panic!(box 413 as Box<dyn std::any::Any + Send>);
}
