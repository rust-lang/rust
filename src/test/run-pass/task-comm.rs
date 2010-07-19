
fn main() -> () {
    // test00(true);
    // test01();
    // test02();
    // test03();
    // test04();
    // test05();
    test06();
}

io fn test00_start(chan[int] ch, int message, int count) {
    log "Starting test00_start";
    let int i = 0;
    while (i < count) {
        log "Sending Message";
        ch <| message;
        i = i + 1;
    }
    log "Ending test00_start";
}

io fn test00(bool is_multithreaded) {
    let int number_of_tasks = 1;
    let int number_of_messages = 64;
    log "Creating tasks";
    
    let port[int] po = port();
    let chan[int] ch = chan(po);
    
    let int i = 0;
    
    let vec[task] tasks = vec();
    while (i < number_of_tasks) {
        i = i + 1;
        if (is_multithreaded) {
            tasks += vec(
                spawn thread test00_start(ch, i, number_of_messages));
        } else {
            tasks += vec(spawn test00_start(ch, i, number_of_messages));
        }
    }
    
    let int sum = 0;
    for (task t in tasks) {
        i = 0;
        while (i < number_of_messages) {
            let int value <- po;
            sum += value;
            i = i + 1;
        }
    }

    for (task t in tasks) {
        join t;
    }
    
    log "Completed: Final number is: ";
    check (sum == number_of_messages * 
           (number_of_tasks * number_of_tasks + number_of_tasks) / 2);
}

io fn test01() {
    let port[int] p = port();
    log "Reading from a port that is never written to.";
    let int value <- p;
    log value;
}

io fn test02() {
    let port[int] p = port();
    let chan[int] c = chan(p);
    log "Writing to a local task channel.";
    c <| 42;
    log "Reading from a local task port.";
    let int value <- p;
    log value;
}

obj vector(mutable int x, int y) {
    fn length() -> int {
        x = x + 2;
        ret x + y;
    }
}

fn test03() {
    log "Creating object ...";
    let mutable vector v = vector(1, 2);
    log "created object ...";
    let mutable vector t = v;
    log v.length();
}

fn test04_start() {
    log "Started Task";
    let int i = 1024 * 1024 * 64;
    while (i > 0) {
        i = i - 1;
    }
    log "Finished Task";
}

fn test04() {
    log "Spawning lots of tasks.";
    let int i = 64;
    while (i > 0) {
        i = i - 1;
        spawn thread test04_start();
    }
    log "Finishing up.";
}

io fn test05_start(chan[int] ch) {
    ch <| 10;
    ch <| 20;
    ch <| 30;
    ch <| 30;
    ch <| 30;    
}

io fn test05() {
    let port[int] po = port();
    let chan[int] ch = chan(po);
    spawn thread test05_start(ch);
    let int value <- po;
    value <- po;
    value <- po;
    log value;
}

fn test06_start(int task_number) {
    log "Started task.";
    let int i = 0;
    while (i < 100000000) {
        i = i + 1;    
    }
    log "Finished task.";
}
    
fn test06() {
    let int number_of_tasks = 32;
    log "Creating tasks";
    
    let int i = 0;
    
    let vec[task] tasks = vec();
    while (i < number_of_tasks) {
        i = i + 1;
        tasks += vec(spawn thread test06_start(i));
        // tasks += vec(spawn test06_start(i));
    }
    
    for (task t in tasks) {
        join t;
    }
}










