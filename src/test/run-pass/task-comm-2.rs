// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
fn main() -> () {    
    log "===== SPAWNING and JOINING TASKS =====";
    test00(false);
    log "===== SPAWNING and JOINING THREAD TASKS =====";
    test00(true);
}

fn start(int task_number) {
    log "Started task.";
    let int i = 0;
    while (i < 10000) {
        i = i + 1;
    }
    log "Finished task.";
}
    
fn test00(bool create_threads) {
    let int number_of_tasks = 8;
    
    let int i = 0;
    let vec[task] tasks = vec();
    while (i < number_of_tasks) {
        i = i + 1;
        if (create_threads) {
            tasks += vec(spawn thread start(i));
        } else {
            tasks += vec(spawn start(i));
        }
    }
    
    for (task t in tasks) {
        join t;
    }
    
    log "Joined all task.";
}