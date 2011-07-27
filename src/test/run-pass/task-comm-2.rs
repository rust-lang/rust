// xfail-stage0

use std;

import std::task;

fn main() { log "===== SPAWNING and JOINING THREAD TASKS ====="; test00(); }

fn start(task_number: int) {
    log "Started task.";
    let i: int = 0;
    while i < 10000 { i = i + 1; }
    log "Finished task.";
}

fn test00() {
    let number_of_tasks: int = 8;

    let i: int = 0;
    let tasks: vec[task] = [];
    while i < number_of_tasks { i = i + 1; tasks += [spawn start(i)]; }

    for t: task  in tasks { task::join(t); }

    log "Joined all task.";
}