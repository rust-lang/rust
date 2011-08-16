use std;

import std::task;
import std::task::task_id;
import std::comm;
import std::comm::_chan;
import std::comm::_port;
import std::comm::send;

fn main() {
    test00();
    // test01();
    test02();
    test03();
    test04();
    test05();
    test06();
}

fn test00_start(ch: _chan[int], message: int, count: int) {
    log "Starting test00_start";
    let i: int = 0;
    while i < count { log "Sending Message"; send(ch, message+0); i = i + 1; }
    log "Ending test00_start";
}

fn test00() {
    let number_of_tasks: int = 1;
    let number_of_messages: int = 4;
    log "Creating tasks";

    let po = comm::mk_port();
    let ch = po.mk_chan();

    let i: int = 0;

    let tasks = [];
    while i < number_of_tasks {
        i = i + 1;
        tasks += [task::_spawn(bind test00_start(ch, i, number_of_messages))];
    }

    let sum: int = 0;
    for t: task_id in tasks {
        i = 0;
        while i < number_of_messages {
            sum += po.recv();
            i = i + 1;
        }
    }

    for t: task_id in tasks { task::join_id(t); }

    log "Completed: Final number is: ";
    assert (sum ==
            number_of_messages *
            (number_of_tasks * number_of_tasks + number_of_tasks) / 2);
}

fn test01() {
    let p = comm::mk_port();
    log "Reading from a port that is never written to.";
    let value: int = p.recv();
    log value;
}

fn test02() {
    let p = comm::mk_port();
    let c = p.mk_chan();
    log "Writing to a local task channel.";
    send(c, 42);
    log "Reading from a local task port.";
    let value: int = p.recv();
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
    while i > 0 { i = i - 1; task::_spawn(bind test04_start()); }
    log "Finishing up.";
}

fn test05_start(ch: _chan[int]) {
    send(ch, 10);
    send(ch, 20);
    send(ch, 30);
    send(ch, 30);
    send(ch, 30);
}

fn test05() {
    let po = comm::mk_port();
    let ch = po.mk_chan();
    task::_spawn(bind test05_start(ch));
    let value: int;
    value = po.recv();
    value = po.recv();
    value = po.recv();
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

    let tasks = [];
    while i < number_of_tasks {
        i = i + 1; tasks += [task::_spawn(bind test06_start(i))]; }


    for t: task_id in tasks { task::join_id(t); }
}










