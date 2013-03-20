type task_id = int;

#[deriving(Eq)]
pub enum Task {
    TaskHandle(task_id)
}

pub fn main() { }
