//@ build-pass
//@ aux-build:issue-97708-aux.rs

extern crate issue_97708_aux;
use issue_97708_aux::TaskStub;

static TASK_STUB: TaskStub = TaskStub::new();

fn main() {}
