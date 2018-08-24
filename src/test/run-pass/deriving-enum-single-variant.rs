// pretty-expanded FIXME #23616

pub type task_id = isize;

#[derive(PartialEq)]
pub enum Task {
    TaskHandle(task_id)
}

pub fn main() { }
