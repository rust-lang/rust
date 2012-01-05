use std;
import task;

fn main() { test00(); }

fn start(&&task_number: int) { #debug("Started / Finished task."); }

fn test00() {
    let i: int = 0;
    let t = task::spawn_joinable {|| start(i); };

    // Sleep long enough for the task to finish.
    task::sleep(10000u);

    // Try joining tasks that have already finished.
    task::join(t);

    #debug("Joined task.");
}
