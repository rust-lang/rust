use std;

fn main() { test00(); }

fn start(&&task_number: int) { debug!("Started / Finished task."); }

fn test00() {
    let i: int = 0;
    let mut result = None;
    do task::task().future_result(|+r| { result = Some(r); }).spawn {
        start(i)
    }

    // Sleep long enough for the task to finish.
    let mut i = 0;
    while i < 10000 {
        task::yield();
        i += 1;
    }

    // Try joining tasks that have already finished.
    future::get(&option::unwrap(result));

    debug!("Joined task.");
}
