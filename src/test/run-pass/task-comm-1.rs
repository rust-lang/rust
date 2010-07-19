fn main() -> () {
   test00(); 
}

fn start() {
    log "Started / Finished Task.";
}

fn test00() {
    let task t = spawn thread start();
    join t;
    log "Completing.";
}