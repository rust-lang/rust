use std;
import std::task;

fn main() { log "===== WITHOUT THREADS ====="; test00(); }

fn test00_start(ch: chan[int], message: int, count: int) {
    log "Starting test00_start";
    let i: int = 0;
    while i < count { log "Sending Message"; ch <| message; i = i + 1; }
    log "Ending test00_start";
}

fn test00() {
    let number_of_tasks: int = 16;
    let number_of_messages: int = 4;

    log "Creating tasks";

    let po: port[int] = port();
    let ch: chan[int] = chan(po);

    let i: int = 0;

    // Create and spawn tasks...
    let tasks: vec[task] = [];
    while i < number_of_tasks {
        tasks += [spawn test00_start(ch, i, number_of_messages)];
        i = i + 1;
    }

    // Read from spawned tasks...
    let sum: int = 0;
    for t: task  in tasks {
        i = 0;
        while i < number_of_messages {
            let value: int;
            po |> value;
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