type task_id = int;

#[deriving_eq]
pub enum Task {
    TaskHandle(task_id)
}

fn main() { }
