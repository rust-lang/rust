// xfail-stage0
// xfail-stage1
// xfail-stage2
fn main() -> () {
   test00(); 
}

fn start() {
    log "Started / Finished task.";
}

fn test00() {
    let task t = spawn thread start();
    join t;
    log "Completing.";
}