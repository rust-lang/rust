// error-pattern:panicked at 'Box<Any>'

#![feature(box_syntax)]

fn main() {
    panic!(box 413 as Box<::std::any::Any + Send>);
}
