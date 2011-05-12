// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
use std;
import std::_task;

fn main() -> () {
    test00();
}

fn start(int task_number) {
    log "Started / Finished task.";
}
    
fn test00() {    
    let int i = 0;
    let task t = spawn thread "child" start(i);
    
    // Sleep long enough for the task to finish.
    _task::sleep(10000u);
    
    // Try joining tasks that have already finished.
    join t;
    
    log "Joined task.";
}