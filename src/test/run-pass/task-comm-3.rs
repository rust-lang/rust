use std;
import std::task;
import std::comm;

fn main() { log "===== WITHOUT THREADS ====="; test00(); }

fn test00_start(pch: *u8, message: int, count: int) {
    log "Starting test00_start";
    let ch = comm::chan_from_unsafe_ptr(pch);
    let i: int = 0;
    while i < count { log "Sending Message"; ch.send(message); i = i + 1; }
    log "Ending test00_start";
}

fn test00() {
    let number_of_tasks: int = 16;
    let number_of_messages: int = 4;

    log "Creating tasks";

    let po = comm::mk_port();
    let ch = po.mk_chan();

    let i: int = 0;

    // Create and spawn tasks...
    let tasks: vec[task] = [];
    while i < number_of_tasks {
        tasks += [spawn test00_start(ch.unsafe_ptr(), i, number_of_messages)];
        i = i + 1;
    }

    // Read from spawned tasks...
    let sum: int = 0;
    for t: task  in tasks {
        i = 0;
        while i < number_of_messages {
            let value: int;
            value = po.recv();
            sum += value;
            i = i + 1;
        }
    }

    // Join spawned tasks...
    for t: task  in tasks { task::join(t); }

    log "Completed: Final number is: ";
    // assert (sum == (((number_of_tasks * (number_of_tasks - 1)) / 2) *
    //       number_of_messages));
    assert (sum == 480);
}
