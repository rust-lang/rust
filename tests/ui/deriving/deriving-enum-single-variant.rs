// run-pass
// pretty-expanded FIXME #23616
#![allow(non_camel_case_types)]

pub type task_id = isize;

#[derive(PartialEq)]
pub enum Task {
    TaskHandle(task_id)
}

pub fn main() { }
