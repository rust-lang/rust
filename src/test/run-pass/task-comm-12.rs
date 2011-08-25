use std;
import std::task;

fn main() { test00(); }

fn start(task_number: int) { log "Started / Finished task."; }

fn test00() {
    let i: int = 0;
    let t = task::spawn_joinable(bind start(i));

    // Sleep long enough for the task to finish.
    task::sleep(10000u);

    // Try joining tasks that have already finished.
    task::join(t);

    log "Joined task.";
}
