use std;

import std::task;

fn main() {
    test00();
    // test01();
    test02();
    test03();
    test04();
    test05();
    test06();
}

fn test00_start(ch: chan[int], message: int, count: int) {
    log "Starting test00_start";
    let i: int = 0;
    while i < count { log "Sending Message"; ch <| message; i = i + 1; }
    log "Ending test00_start";
}

fn test00() {
    let number_of_tasks: int = 1;
    let number_of_messages: int = 4;
    log "Creating tasks";

    let po: port[int] = port();
    let ch: chan[int] = chan(po);

    let i: int = 0;

    let tasks: vec[task] = [];
    while i < number_of_tasks {
        i = i + 1;
        tasks += [spawn test00_start(ch, i, number_of_messages)];
    }

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

    for t: task  in tasks { task::join(t); }

    log "Completed: Final number is: ";
    assert (sum ==
                number_of_messages *
                    (number_of_tasks * number_of_tasks + number_of_tasks) /
                    2);
}

fn test01() {
    let p: port[int] = port();
    log "Reading from a port that is never written to.";
    let value: int;
    p |> value;
    log value;
}

fn test02() {
    let p: port[int] = port();
    let c: chan[int] = chan(p);
    log "Writing to a local task channel.";
    c <| 42;
    log "Reading from a local task port.";
    let value: int;
    p |> value;
    log value;
}

obj vector(mutable x: int, y: int) {
    fn length() -> int { x = x + 2; ret x + y; }
}

fn test03() {
    log "Creating object ...";
    let v: vector = vector(1, 2);
    log "created object ...";
    let t: vector = v;
    log v.length();
}

fn test04_start() {
    log "Started task";
    let i: int = 1024 * 1024 * 64;
    while i > 0 { i = i - 1; }
    log "Finished task";
}

fn test04() {
    log "Spawning lots of tasks.";
    let i: int = 4;
    while i > 0 { i = i - 1; spawn test04_start(); }
    log "Finishing up.";
}

fn test05_start(ch: chan[int]) {
    ch <| 10;
    ch <| 20;
    ch <| 30;
    ch <| 30;
    ch <| 30;
}

fn test05() {
    let po: port[int] = port();
    let ch: chan[int] = chan(po);
    spawn test05_start(ch);
    let value: int;
    po |> value;
    po |> value;
    po |> value;
    log value;
}

fn test06_start(task_number: int) {
    log "Started task.";
    let i: int = 0;
    while i < 100000000 { i = i + 1; }
    log "Finished task.";
}

fn test06() {
    let number_of_tasks: int = 4;
    log "Creating tasks";

    let i: int = 0;

    let tasks: vec[task] = [];
    while i < number_of_tasks { i = i + 1; tasks += [spawn test06_start(i)]; }


    for t: task  in tasks { task::join(t); }
}










