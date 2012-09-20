/*!

The task interface to the runtime

*/

#[doc(hidden)]; // FIXME #3538

#[allow(non_camel_case_types)] // runtime type
type sched_id = int;
#[allow(non_camel_case_types)] // runtime type
type task_id = int;

// These are both opaque runtime/compiler types that we don't know the
// structure of and should only deal with via unsafe pointer
#[allow(non_camel_case_types)] // runtime type
type rust_task = libc::c_void;
#[allow(non_camel_case_types)] // runtime type
type rust_closure = libc::c_void;

extern {
    #[rust_stack]
    fn rust_task_yield(task: *rust_task) -> bool;

    fn rust_get_sched_id() -> sched_id;
    fn rust_new_sched(num_threads: libc::uintptr_t) -> sched_id;
    fn rust_sched_threads() -> libc::size_t;
    fn rust_sched_current_nonlazy_threads() -> libc::size_t;
    fn rust_num_threads() -> libc::uintptr_t;

    fn get_task_id() -> task_id;
    #[rust_stack]
    fn rust_get_task() -> *rust_task;

    fn new_task() -> *rust_task;
    fn rust_new_task_in_sched(id: sched_id) -> *rust_task;

    fn start_task(task: *rust_task, closure: *rust_closure);

    fn rust_task_is_unwinding(task: *rust_task) -> bool;
    fn rust_osmain_sched_id() -> sched_id;
    #[rust_stack]
    fn rust_task_inhibit_kill(t: *rust_task);
    #[rust_stack]
    fn rust_task_allow_kill(t: *rust_task);
    #[rust_stack]
    fn rust_task_inhibit_yield(t: *rust_task);
    #[rust_stack]
    fn rust_task_allow_yield(t: *rust_task);
    fn rust_task_kill_other(task: *rust_task);
    fn rust_task_kill_all(task: *rust_task);

    #[rust_stack]
    fn rust_get_task_local_data(task: *rust_task) -> *libc::c_void;
    #[rust_stack]
    fn rust_set_task_local_data(task: *rust_task, map: *libc::c_void);
    #[rust_stack]
    fn rust_task_local_data_atexit(task: *rust_task, cleanup_fn: *u8);
}
