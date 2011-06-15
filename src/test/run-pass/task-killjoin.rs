// xfail-stage0
// xfail-stage1
// xfail-stage2
// Create a task that is supervised by another task,
// join the supervised task from the supervising task,
// then fail the supervised task. The supervised task
// will kill the supervising task, waking it up. The
// supervising task no longer needs to be wakened when
// the supervised task exits.

fn supervised() {
    // Yield to make sure the supervisor joins before we
    // fail. This is currently not needed because the supervisor
    // runs first, but I can imagine that changing.
    yield;
    fail;
}

fn supervisor() {
    let task t = spawn "supervised" supervised();
    join t;
}

fn main() {
    // Start the test in another domain so that
    // the process doesn't return a failure status as a result
    // of the main task being killed.
    let task dom2 = spawn thread "supervisor" supervisor();
    join dom2;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
