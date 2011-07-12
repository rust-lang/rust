// xfail-stage0

use std;
import std::task;

fn main() -> () {
    test00();
}

fn start(int task_number) {
    log "Started / Finished task.";
}
    
fn test00() {    
    let int i = 0;
    let task t = spawn start(i);
    
    // Sleep long enough for the task to finish.
    task::sleep(10000u);
    
    // Try joining tasks that have already finished.
    task::join(t);
    
    log "Joined task.";
}