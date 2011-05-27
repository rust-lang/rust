// xfail-stage0
// xfail-stage1
// xfail-stage2
fn main() -> () {
   log "===== WITHOUT THREADS =====";
   test00(false);
   log "====== WITH THREADS ======";
   test00(true);
}

fn test00_start(chan[int] ch, int message, int count) {
    log "Starting test00_start";
    let int i = 0;
    while (i < count) {
        log "Sending Message";
        ch <| message;
        i = i + 1;
    }
    log "Ending test00_start";
}

fn test00(bool is_multithreaded) {
    let int number_of_tasks = 16;
    let int number_of_messages = 4;
    
    log "Creating tasks";
    
    let port[int] po = port();
    let chan[int] ch = chan(po);
    
    let int i = 0;
    
    // Create and spawn tasks...
    let vec[task] tasks = [];
    while (i < number_of_tasks) {
        if (is_multithreaded) {
            tasks += [
                spawn thread test00_start(ch, i, number_of_messages)];
        } else {
            tasks += [spawn test00_start(ch, i, number_of_messages)];
        }
        i = i + 1;
    }
    
    // Read from spawned tasks...
    let int sum = 0;
    for (task t in tasks) {
        i = 0;
        while (i < number_of_messages) {
            let int value; po |> value;
            sum += value;
            i = i + 1;
        }
    }

    // Join spawned tasks...
    for (task t in tasks) {
        join t;
    }
    
    log "Completed: Final number is: ";
    // assert (sum == (((number_of_tasks * (number_of_tasks - 1)) / 2) * 
    //       number_of_messages));
    assert (sum == 480);
}
