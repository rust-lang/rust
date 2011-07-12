// xfail-stage0

use std;

import std::task;

fn main() -> () {    
    log "===== SPAWNING and JOINING THREAD TASKS =====";
    test00();
}

fn start(int task_number) {
    log "Started task.";
    let int i = 0;
    while (i < 10000) {
        i = i + 1;
    }
    log "Finished task.";
}
    
fn test00() {
    let int number_of_tasks = 8;
    
    let int i = 0;
    let vec[task] tasks = [];
    while (i < number_of_tasks) {
        i = i + 1;
        tasks += [spawn start(i)];
    }
    
    for (task t in tasks) {
        task::join(t);
    }
    
    log "Joined all task.";
}