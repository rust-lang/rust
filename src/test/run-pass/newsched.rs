// compile-flags: --test

// FIXME #4830 uncomment `priv` field modifiers

extern mod std;

use core::io::println;
use core::private::run_in_bare_thread;
use core::cast::{transmute, transmute_mut_region};
use uv_sched_event_loop::UvSchedEventLoop;
use core::libc::{uintptr_t, c_void};
use core::ptr::mut_null;
use context::Context;

fn macros() {
    macro_rules! rtdebug (
        ($( $arg:expr),+) => ( { } )
    )

    macro_rules! rtdebug_ (
        ($( $arg:expr),+) => ( {
            dumb_println(fmt!( $($arg),+ ));

            fn dumb_println(s: &str) {
                use core::str::as_c_str;
                use core::libc::c_char;

                extern {
                    fn printf(s: *c_char);
                }

                do as_c_str(s.to_str() + "\n") |s| {
                    unsafe { printf(s); }
                }
            }

        } )
    )
}

/// The Scheduler is responsible for coordinating execution of Tasks
/// on a single thread. When the scheduler is running it is owned by
/// thread local storage and the running task is owned by the
/// scheduler.
pub struct Scheduler {
    task_queue: WorkQueue<~Task>,
    stack_pool: StackPool,
    /// The event loop used to drive the scheduler and perform I/O
    /// NOTE: This should be ~SchedEventLoop
    /*priv*/ event_loop: ~UvSchedEventLoop,
    /// The scheduler's saved context.
    /// Always valid when a task is executing, otherwise not
    /*priv*/ saved_context: Context,
    /// The currently executing task
    /*priv*/ current_task: Option<~Task>,
    /// A queue of jobs to perform immediately upon return from task
    /// context to scheduler context.
    /*priv*/ cleanup_jobs: ~[CleanupJob]
}

enum CleanupJob {
    RescheduleTask(~Task),
    RecycleTask(~Task)
}

impl Scheduler {

    static fn new(event_loop: ~UvSchedEventLoop) -> Scheduler {
        Scheduler {
            event_loop: event_loop,
            task_queue: WorkQueue::new(),
            stack_pool: StackPool::new(),
            saved_context: Context::empty(),
            current_task: None,
            cleanup_jobs: ~[]
        }
    }
    
    // NOTE: This may eventually need to be refactored so that
    // the scheduler itself doesn't have to call event_loop.run.
    // That will be important for embedding the runtime into external
    // event loops.
    fn run(~self) -> ~Scheduler {
        assert !self.in_task_context();

        // Give ownership of the scheduler (self) to the thread
        do install_thread_local_scheduler(self) |scheduler| {
            fn run_scheduler_once() {
                do use_thread_local_scheduler |scheduler| {
                    if scheduler.resume_task_from_queue() {
                        // Ok, a task ran. Nice! We'll do it again later
                        scheduler.event_loop.callback(run_scheduler_once);
                    }
                }
            }

            scheduler.event_loop.callback(run_scheduler_once);
            scheduler.event_loop.run();
        }
    }


    // * Scheduler-context operations

    fn resume_task_from_queue(&mut self) -> bool {
        assert !self.in_task_context();

        let mut self = self;
        match self.task_queue.pop_front() {
            Some(task) => {
                self.resume_task_immediately(task);
                return true;
            }
            None => {
                rtdebug!("no tasks in queue");
                return false;
            }
        }
    }

    fn resume_task_immediately(&mut self, task: ~Task) {
        assert !self.in_task_context();

        rtdebug!("scheduling a task");

        // Store the task in the scheduler so it can be grabbed later
        self.current_task = Some(task);
        self.swap_in_task();

        // Running tasks may have asked us to do some cleanup
        self.run_cleanup_jobs();
    }


    // * Task-context operations

    fn terminate_current_task(&mut self) {
        assert self.in_task_context();

        rtdebug!("ending running task");

        let dead_task = self.current_task.swap_unwrap();
        self.enqueue_cleanup_job(RecycleTask(dead_task));
        let recycle_job = self.last_cleanup_job();
        let dead_task: &mut Task = match recycle_job {
            &RecycleTask(~ref mut task) => task, _ => fail!()
        };

        self.swap_out_task(dead_task);
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this.
    fn resume_task_from_running_task_direct(&mut self, next_task: ~Task) {
        assert self.in_task_context();

        rtdebug!("switching tasks");

        let old_running_task = self.current_task.swap_unwrap();
        self.enqueue_cleanup_job(RescheduleTask(old_running_task));
        let reschedule_job = self.last_cleanup_job();
        let old_running_task: &mut Task = match reschedule_job {
            &RescheduleTask(~ref mut task) => task, _ => fail!()
        };

        self.current_task = Some(next_task);
        self.swap_in_task_from_running_task(old_running_task);
    }


    // * Context switching

    // NB: When switching to a task callers are expected to first set self.running_task
    // When switching away from a task likewise move out of the self.running_task

    priv fn swap_in_task(&mut self) {
        // Take pointers to both the task and scheduler's saved registers.
        let running_task: &~Task = self.current_task.get_ref();
        let task_context = &running_task.saved_context;
        let scheduler_context = &mut self.saved_context;

        // Context switch to the task, restoring it's registers
        // and saving the scheduler's
        Context::swap(scheduler_context, task_context);
    }

    priv fn swap_out_task(&mut self, running_task: &mut Task) {
        let task_context = &mut running_task.saved_context;
        let scheduler_context = &self.saved_context;
        Context::swap(task_context, scheduler_context);
    }

    priv fn swap_in_task_from_running_task(&mut self, running_task: &mut Task) {
        let running_task_context = &mut running_task.saved_context;
        let next_context = &self.current_task.get_ref().saved_context;
        Context::swap(running_task_context, next_context);
    }


    // * Other stuff

    fn in_task_context(&self) -> bool { self.current_task.is_some() }

    fn enqueue_cleanup_job(&mut self, job: CleanupJob) {
        self.cleanup_jobs.unshift(job);
    }

    fn run_cleanup_jobs(&mut self) {
        assert !self.in_task_context();
        rtdebug!("running cleanup jobs");

        while !self.cleanup_jobs.is_empty() {
            match self.cleanup_jobs.pop() {
                RescheduleTask(task) => {
                    // NB: Pushing to the *front* of the queue
                    self.task_queue.push_front(task);
                }
                RecycleTask(task) => task.recycle(&mut self.stack_pool),
            }
        }
    }

    fn last_cleanup_job(&mut self) -> &self/mut CleanupJob {
        assert !self.cleanup_jobs.is_empty();
        &mut self.cleanup_jobs[0]
    }
}

const TASK_MIN_STACK_SIZE: uint = 10000000; // XXX: Too much stack

struct Task {
    /// The task entry point, saved here for later destruction
    /*priv*/ start: ~~fn(),
    /// The segment of stack on which the task is currently running or,
    /// if the task is blocked, on which the task will resume execution
    /*priv*/ current_stack_segment: StackSegment,
    /// These are always valid when the task is not running, unless the task is dead
    /*priv*/ saved_context: Context,
}

impl Task {
    static fn new(stack_pool: &mut StackPool, start: ~fn()) -> Task {
        // NOTE: Putting main into a ~ so it's a thin pointer and can be passed to the spawn function.
        // Another unfortunate allocation
        let start = ~Task::build_start_wrapper(start);
        let mut stack = stack_pool.take_segment(TASK_MIN_STACK_SIZE);
        // NB: Context holds a pointer to that ~fn
        let initial_context = Context::new(&*start, &mut stack);
        return Task {
            start: start,
            current_stack_segment: stack,
            saved_context: initial_context,
        };
    }

    static priv fn build_start_wrapper(start: ~fn()) -> ~fn() {
        // NOTE: The old code didn't have this extra allocation
        let wrapper: ~fn() = || {
            start();

            let mut sched = ThreadLocalScheduler::new();
            let sched = sched.get_scheduler();
            sched.terminate_current_task();
        };
        return wrapper;
    }

    /// Destroy the task and try to reuse its components
    fn recycle(~self, stack_pool: &mut StackPool) {
        match self {
            ~Task {current_stack_segment, _} => {
                stack_pool.give_segment(current_stack_segment);
            }
        }
    }
}

#[test]
fn test_simple_scheduling() {
    do run_in_bare_thread {
        let mut task_ran = false;
        let task_ran_ptr: *mut bool = &mut task_ran;

        let mut sched = ~UvSchedEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            unsafe { *task_ran_ptr = true; }
        };
        sched.task_queue.push_back(task);
        sched.run();
        assert task_ran;
    }
}

#[test]
fn test_several_tasks() {
    do run_in_bare_thread {
        let total = 10;
        let mut task_count = 0;
        let task_count_ptr: *mut int = &mut task_count;

        let mut sched = ~UvSchedEventLoop::new_scheduler();
        for int::range(0, total) |_| {
            let task = ~do Task::new(&mut sched.stack_pool) {
                unsafe { *task_count_ptr = *task_count_ptr + 1; }
            };
            sched.task_queue.push_back(task);
        }
        sched.run();
        assert task_count == total;
    }
}

#[test]
fn test_swap_tasks() {
    do run_in_bare_thread {
        let mut count = 0;
        let count_ptr: *mut int = &mut count;

        let mut sched = ~UvSchedEventLoop::new_scheduler();
        let task1 = ~do Task::new(&mut sched.stack_pool) {
            unsafe { *count_ptr = *count_ptr + 1; }
            do use_thread_local_scheduler |sched| {
                let task2 = ~do Task::new(&mut sched.stack_pool) {
                    unsafe { *count_ptr = *count_ptr + 1; }
                };
                // Context switch directly to the new task
                sched.resume_task_from_running_task_direct(task2);
            }
            unsafe { *count_ptr = *count_ptr + 1; }
        };
        sched.task_queue.push_back(task1);
        sched.run();
        assert count == 3;
    }
}

#[bench] #[test] #[ignore(reason = "long test")]
fn test_run_a_lot_of_tasks_queued() {
    do run_in_bare_thread {
        const MAX: int = 1000000;
        let mut count = 0;
        let count_ptr: *mut int = &mut count;

        let mut sched = ~UvSchedEventLoop::new_scheduler();

        let start_task = ~do Task::new(&mut sched.stack_pool) {
            run_task(count_ptr);
        };
        sched.task_queue.push_back(start_task);
        sched.run();

        assert count == MAX;

        fn run_task(count_ptr: *mut int) {
            do use_thread_local_scheduler |sched| {
                let task = ~do Task::new(&mut sched.stack_pool) {
                    unsafe {
                        *count_ptr = *count_ptr + 1;
                        if *count_ptr != MAX {
                            run_task(count_ptr);
                        }
                    }
                };
                sched.task_queue.push_back(task);
            }
        };
    }
}

#[bench] #[test] #[ignore(reason = "too much stack allocation")]
fn test_run_a_lot_of_tasks_direct() {
    do run_in_bare_thread {
        const MAX: int = 100000;
        let mut count = 0;
        let count_ptr: *mut int = &mut count;

        let mut sched = ~UvSchedEventLoop::new_scheduler();

        let start_task = ~do Task::new(&mut sched.stack_pool) {
            run_task(count_ptr);
        };
        sched.task_queue.push_back(start_task);
        sched.run();

        assert count == MAX;

        fn run_task(count_ptr: *mut int) {
            do use_thread_local_scheduler |sched| {
                let task = ~do Task::new(&mut sched.stack_pool) {
                    unsafe {
                        *count_ptr = *count_ptr + 1;
                        if *count_ptr != MAX {
                            run_task(count_ptr);
                        }
                    }
                };
                // Context switch directly to the new task
                sched.resume_task_from_running_task_direct(task);
            }
        };
    }
}

fn install_thread_local_scheduler(scheduler: ~Scheduler, f: &fn(&mut Scheduler)) -> ~Scheduler {
    let mut tlsched = ThreadLocalScheduler::new();
    tlsched.put_scheduler(scheduler);
    {
        let sched = tlsched.get_scheduler();
        f(sched);
    }
    return tlsched.take_scheduler();
}

fn use_thread_local_scheduler(f: &fn(&mut Scheduler)) {
    let mut tlsched = ThreadLocalScheduler::new();
    f(tlsched.get_scheduler());
}

// NB: This is a type so we can use make use of the &self region.
// TODO: Test and describe how this uses &mut self
struct ThreadLocalScheduler(thread_local_storage::Key);

impl ThreadLocalScheduler {
    static fn new() -> ThreadLocalScheduler {
        unsafe {
            // NB: This assumes that the TLS key has been created prior.
            // Currently done in Rust start.
            let key: *mut c_void = rust_get_sched_tls_key();
            let key: &mut thread_local_storage::Key = transmute(key);
            ThreadLocalScheduler(*key)
        }
    }

    fn put_scheduler(&mut self, scheduler: ~Scheduler) {
        unsafe {
            let key = match self { &ThreadLocalScheduler(key) => key };
            let value: *mut c_void = transmute::<~Scheduler, *mut c_void>(scheduler);
            thread_local_storage::set(key, value);
        }
    }

    fn get_scheduler(&mut self) -> &self/mut Scheduler {
        unsafe {
            let key = match self { &ThreadLocalScheduler(key) => key };
            let mut value: *mut c_void = thread_local_storage::get(key);
            assert value.is_not_null();
            {
                let value_ptr = &mut value;
                let sched: &mut ~Scheduler = transmute::<&mut *mut c_void, &mut ~Scheduler>(value_ptr);
                let sched: &mut Scheduler = &mut **sched;
                return sched;
            }
        }
    }

    fn take_scheduler(&mut self) -> ~Scheduler {
        unsafe {
            let key = match self { &ThreadLocalScheduler(key) => key };
            let value: *mut c_void = thread_local_storage::get(key);
            assert value.is_not_null();
            let sched = transmute(value);
            thread_local_storage::set(key, mut_null());
            return sched;
        }
    }
}

extern {
    fn rust_get_sched_tls_key() -> *mut c_void;
}

#[test]
fn thread_local_scheduler_smoke_test() {
    let scheduler = ~UvSchedEventLoop::new_scheduler();
    let mut tls_scheduler = ThreadLocalScheduler::new();
    tls_scheduler.put_scheduler(scheduler);
    {
        let _scheduler = tls_scheduler.get_scheduler();
    }
    let _scheduler = tls_scheduler.take_scheduler();
}

#[test]
fn thread_local_scheduler_two_instances() {
    let scheduler = ~UvSchedEventLoop::new_scheduler();
    let mut tls_scheduler = ThreadLocalScheduler::new();
    tls_scheduler.put_scheduler(scheduler);
    {

        let _scheduler = tls_scheduler.get_scheduler();
    }
    {
        let scheduler = tls_scheduler.take_scheduler();
        tls_scheduler.put_scheduler(scheduler);
    }

    let mut tls_scheduler = ThreadLocalScheduler::new();
    {
        let _scheduler = tls_scheduler.get_scheduler();
    }
    let _scheduler = tls_scheduler.take_scheduler();
}

mod thread_local_storage {

    use core::libc::{c_uint, c_int, c_void};
    use core::ptr::null;

    #[cfg(unix)]
    pub type Key = pthread_key_t;

    #[cfg(unix)]
    pub unsafe fn create(key: &mut Key) {
        unsafe { assert 0 == pthread_key_create(key, null()); }
    }

    #[cfg(unix)]
    pub unsafe fn set(key: Key, value: *mut c_void) {
        unsafe { assert 0 == pthread_setspecific(key, value); }
    }

    #[cfg(unix)]
    pub unsafe fn get(key: Key) -> *mut c_void {
        unsafe { pthread_getspecific(key) }
    }

    #[cfg(unix)]
    type pthread_key_t = c_uint;

    #[cfg(unix)]
    extern {
        fn pthread_key_create(key: *mut pthread_key_t, dtor: *u8) -> c_int;
        fn pthread_setspecific(key: pthread_key_t, value: *mut c_void) -> c_int;
        fn pthread_getspecific(key: pthread_key_t) -> *mut c_void;
    }

    #[test]
    fn tls_smoke_test() {
        use core::cast::transmute;
        unsafe {
            let mut key = 0;
            let value = ~20;
            create(&mut key);
            set(key, transmute(value));
            let value: ~int = transmute(get(key));
            assert value == ~20;
            let value = ~30;
            set(key, transmute(value));
            let value: ~int = transmute(get(key));
            assert value == ~30;
        }
    }
}

struct WorkQueue<T> {
    priv queue: ~[T]
}

impl<T> WorkQueue<T> {
    static fn new() -> WorkQueue<T> {
        WorkQueue {
            queue: ~[]
        }
    }

    fn push_back(&mut self, value: T) {
        self.queue.push(value)
    }

    fn pop_back(&mut self) -> Option<T> {
        if !self.queue.is_empty() {
            Some(self.queue.pop())
        } else {
            None
        }
    }

    fn push_front(&mut self, value: T) {
        self.queue.unshift(value)
    }

    fn pop_front(&mut self) -> Option<T> {
        if !self.queue.is_empty() {
            Some(self.queue.shift())
        } else {
            None
        }
    }
}

struct StackSegment {
    buf: ~[u8]
}

impl StackSegment {
    static fn new(size: uint) -> StackSegment {
        // Crate a block of unallocated values
        let mut stack = vec::with_capacity(size);
        unsafe {
            vec::raw::set_len(&mut stack, size);
        }

        StackSegment {
            buf: stack
        }
    }

    fn end(&self) -> *uint {
        unsafe {
            vec::raw::to_ptr(self.buf).offset(self.buf.len()) as *uint
        }
    }
}

struct StackPool(());

impl StackPool {

    static fn new() -> StackPool { StackPool(()) }

    fn take_segment(&self, min_size: uint) -> StackSegment {
        StackSegment::new(min_size)
    }

    fn give_segment(&self, _stack: StackSegment) {
    }
}

mod context {
    use super::StackSegment;
    use core::libc::c_void;
    use core::cast::{transmute, transmute_mut_unsafe, transmute_region, transmute_mut_region};

    // NOTE: Registers is boxed so that it is 16-byte aligned, for storing SSE regs.
    // It would be marginally better not to do this. In C++ we use an attribute on a struct.
    pub struct Context(~Registers);

    pub impl Context {
        static fn empty() -> Context {
            Context(new_regs())
        }

        /// Create a new context that will resume execution by running ~fn()
        /// # Safety Note
        /// The `start` closure must remain valid for the life of the Task
        static fn new(start: &~fn(), stack: &mut StackSegment) -> Context {

            // The C-ABI function that is the task entry point
            extern fn task_start_wrapper(f: &~fn()) { (*f)() }

            let fp: *c_void = task_start_wrapper as *c_void;
            let argp: *c_void = unsafe { transmute::<&~fn(), *c_void>(&*start) };
            let sp: *uint = stack.end();

            // Save and then immediately load the current context,
            // which we will then modify to call the given function when restored
            let mut regs = new_regs();
            unsafe { swap_registers(transmute_mut_region(&mut *regs), transmute_region(&*regs)) };

            initialize_call_frame(&mut *regs, fp, argp, sp);

            return Context(regs);
        }

        static fn swap(out_context: &mut Context, in_context: &Context) {
            let out_regs: &mut Registers = match out_context { &Context(~ref mut r) => r };
            let in_regs: &Registers = match in_context { &Context(~ref r) => r };

            unsafe { swap_registers(out_regs, in_regs) };
        }
    }

    extern {
        fn swap_registers(out_regs: *mut Registers, in_regs: *Registers);
    }

    // Definitions of these registers is in rt/arch/x86_64/regs.h
    #[cfg(target_arch = "x86_64")]
    type Registers = [uint * 22];

    #[cfg(target_arch = "x86_64")]
    fn new_regs() -> ~Registers { ~[0, .. 22] }

    #[cfg(target_arch = "x86_64")]
    fn initialize_call_frame(regs: &mut Registers,
                             fptr: *c_void, arg: *c_void, sp: *uint) {

        // Redefinitions from regs.h
        const RUSTRT_ARG0: uint = 3;
        const RUSTRT_RSP: uint = 1;
        const RUSTRT_IP: uint = 8;
        const RUSTRT_RBP: uint = 2;

        let sp = sp as *uint;
        let sp = align_down(sp);
        let sp = sp.offset(-1);
        let sp = unsafe { transmute_mut_unsafe(sp) };

        // The final return address. 0 indicates the bottom of the stack
        unsafe { *sp = 0; }

        rtdebug!("creating call frame");
        rtdebug!("fptr %x", fptr as uint);
        rtdebug!("arg %x", arg as uint);
        rtdebug!("sp %x", sp as uint);

        regs[RUSTRT_ARG0] = arg as uint;
        regs[RUSTRT_RSP] = sp as uint;
        regs[RUSTRT_IP] = fptr as uint;

        // Last base pointer on the stack should be 0
        regs[RUSTRT_RBP] = 0;
    }

    fn align_down(sp: *uint) -> *uint {
        unsafe {
            let sp = transmute::<*uint, uint>(sp);
            let sp = sp & !(16 - 1);
            transmute::<uint, *uint>(sp)
        }
    }

    // NOTE: ptr::offset is positive ints only
    #[inline(always)]
    pub pure fn offset<T>(ptr: *T, count: int) -> *T {
        use core::sys::size_of;
        unsafe {
            (ptr as int + count * (size_of::<T>() as int)) as *T
        }
    }

}

pub trait SchedEventLoop {
    fn run(&mut self);
    fn callback(&mut self, ~fn());
}

mod uv_sched_event_loop {
    use core::io::println;

    use super::SchedEventLoop;
    use super::Scheduler;
    use super::uv::*;

    pub struct UvSchedEventLoop(Loop);

    pub impl UvSchedEventLoop {
        static fn new() -> UvSchedEventLoop {
            UvSchedEventLoop(Loop::new())
        }

        /// A convenience constructor
        static fn new_scheduler() -> Scheduler {
            Scheduler::new(~UvSchedEventLoop::new())
        }

        priv fn uv_loop(&mut self) -> &self/mut Loop {
            match self { &UvSchedEventLoop(ref mut ptr) => ptr }
        }
    }

    pub impl UvSchedEventLoop: SchedEventLoop {

        fn run(&mut self) {
            self.uv_loop().run();
        }

        fn callback(&mut self, f: ~fn()) {
            let mut idle_watcher =  IdleWatcher::new(self.uv_loop());
            do idle_watcher.start |idle_watcher, status| {
                assert status.is_none();
                let mut idle_watcher = idle_watcher;
                idle_watcher.stop();
                idle_watcher.close();
                f();
            }
        }
    }

    #[test]
    fn test_callback_run_once() {
        do run_in_bare_thread {
            let mut event_loop = UvSchedEventLoop::new();
            let mut count = 0;
            let count_ptr: *mut int = &mut count;
            do event_loop.callback {
                unsafe { *count_ptr += 1 }
            }
            event_loop.run();
            assert count == 1;
        }
    }
}

pub mod uv {
    use core::private::run_in_bare_thread;
    use core::str::raw::from_c_str;
    use core::libc::{c_void, c_int};
    use core::cast::transmute;
    use core::ptr::null;
    use uvll = std::uv::ll;

    /// A trait for callbacks to implement. Provides a little extra type safety
    /// for generic interop functions like `set_watcher_callback`.
    trait Callback { }

    struct UvError(uvll::uv_err_t);

    impl UvError {

        pure fn name(&self) -> ~str {
            unsafe { 
                let inner = match self { &UvError(ref a) => a };
                let name_str = uvll::err_name(inner);
                assert name_str.is_not_null();
                from_c_str(name_str)
            }
        }

        pure fn desc(&self) -> ~str {
            unsafe {
                let inner = match self { &UvError(ref a) => a };
                let desc_str = uvll::strerror(inner);
                assert desc_str.is_not_null();
                from_c_str(desc_str)
            }
        }
    }

    impl UvError: ToStr {
        pure fn to_str(&self) -> ~str {
            fmt!("%s: %s", self.name(), self.desc())
        }
    }

    #[test]
    fn error_smoke_test() {
        let err = uvll::uv_err_t { code: 1, sys_errno_: 1 };
        let err: UvError = UvError(err);
        assert err.to_str() == ~"EOF: end of file";
    }

    /// A trait for types that wrap a native handle
    trait NativeHandle<T> {
        pub fn native_handle(&self) -> T;
    }

    /// The uv event loop. The event loop is an owned type that is deleted
    /// once it goes out of scope.
    /// XXX: Loop(*handle) is buggy with destructors. Normal structs
    /// with dtors may not be destructured, but tuple structs can,
    /// but the results are not correct.
    pub struct Loop {
        handle: *uvll::uv_loop_t
    }

    pub impl Loop {
        static fn new() -> Loop {
            let handle = unsafe { uvll::loop_new() };
            assert handle.is_not_null();
            Loop { handle: handle }
        }

        fn run(&mut self) {
            unsafe { uvll::run(self.native_handle()) };
        }
    }

    pub impl Loop: NativeHandle<*uvll::uv_loop_t> {
        fn native_handle(&self) -> *uvll::uv_loop_t {
            self.handle
        }
    }

    pub impl Loop: Drop {
        fn finalize(&self) {
            unsafe { uvll::loop_delete(self.native_handle()) };
        }
    }

    #[test]
    fn loop_smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            loop_.run();
        }
    }

    /// The trait implemented by uv 'watchers' (handles). Watchers
    /// are non-owning wrappers around the uv handles and are not completely safe -
    /// there may be multiple instances for a single underlying handle.
    /// Watchers are generally created, then `start`ed, `stop`ed and `close`ed,
    /// but due to their complex life cycle may not be entirely memory safe
    /// if used in unanticipated patterns.
    trait Watcher { }

    pub struct IdleWatcher(*uvll::uv_idle_t);

    impl IdleWatcher: Watcher { }

    type IdleCallback = ~fn(IdleWatcher, Option<UvError>);
    impl IdleCallback: Callback { }

    pub impl IdleWatcher {
        static fn new(loop_: &mut Loop) -> IdleWatcher {
            unsafe {
                let handle = uvll::idle_new();
                assert handle.is_not_null();
                assert 0 == uvll::idle_init(loop_.native_handle(), handle);
                uvll::set_data_for_uv_handle(handle, null::<()>());
                IdleWatcher(handle)
            }
        }

        fn start(&mut self, cb: IdleCallback) {

            set_watcher_callback(self, cb);
            unsafe { assert 0 == uvll::idle_start(self.native_handle(), idle_cb) };

            extern fn idle_cb(handle: *uvll::uv_idle_t, status: c_int) {
                let idle_watcher = IdleWatcher(handle);
                let cb: &IdleCallback = callback_from_watcher(&idle_watcher);
                let status = status_to_maybe_uv_error(handle, status);
                (*cb)(idle_watcher, status);
            }
        }

        fn stop(&mut self) {
            unsafe { assert 0 == uvll::idle_stop(self.native_handle()); }
        }

        fn close(self) {
            unsafe { uvll::close(self.native_handle(), close_cb) };

            extern fn close_cb(handle: *uvll::uv_idle_t) {
                let mut idle_watcher = IdleWatcher(handle);
                drop_watcher_callback::<uvll::uv_idle_t, IdleWatcher, IdleCallback>(&mut idle_watcher);
                unsafe { uvll::idle_delete(handle) };
            }
        }
    }

    pub impl IdleWatcher: NativeHandle<*uvll::uv_idle_t> {
        fn native_handle(&self) -> *uvll::uv_idle_t {
            match self { &IdleWatcher(ptr) => ptr }
        }
    }

    #[test]
    #[ignore(reason = "valgrind - loop destroyed before watcher?")]
    fn idle_new_then_close() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
            idle_watcher.close();
        }
    }

    #[test]
    fn idle_smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
            let mut count = 10;
            let count_ptr: *mut int = &mut count;
            do idle_watcher.start |idle_watcher, status| {
                let mut idle_watcher = idle_watcher;
                assert status.is_none();
                if unsafe { *count_ptr == 10 } {
                    idle_watcher.stop();
                    idle_watcher.close();
                } else {
                    unsafe { *count_ptr = *count_ptr + 1; }
                }
            }
            loop_.run();
            assert count == 10;
        }
    }

    #[test]
    fn idle_start_stop_start() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
            do idle_watcher.start |idle_watcher, status| {
                let mut idle_watcher = idle_watcher;
                assert status.is_none();
                idle_watcher.stop();
                do idle_watcher.start |idle_watcher, status| {
                    let mut idle_watcher = idle_watcher;
                    assert status.is_none();
                    idle_watcher.stop();
                    idle_watcher.close();
                }
            }
            loop_.run();
        }
    }

    fn status_to_maybe_uv_error<T>(handle: *T, status: c_int) -> Option<UvError> {
        if status == 0 {
            None
        } else {
            unsafe {
                let loop_ = uvll::get_loop_for_uv_handle(handle);
                let err = uvll::last_error(loop_);
                Some(UvError(err))
            }
        }
    }

    fn set_watcher_callback<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &mut W, cb: CB) {
        drop_watcher_callback::<H, W, CB>(watcher);
        // NOTE: Boxing the callback so it fits into a pointer. Unfortunate extra allocation
        let boxed_cb = ~cb;
        let data = unsafe { transmute::<~CB, *c_void>(boxed_cb) };
        unsafe { uvll::set_data_for_uv_handle(watcher.native_handle(), data) };
    }

    fn drop_watcher_callback<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &mut W) {
        unsafe {
            let handle = watcher.native_handle();
            let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
            if handle_data.is_not_null() {
                // Take ownership of the callback and drop it
                let _cb = transmute::<*c_void, ~CB>(handle_data);
                // Make sure the pointer is zeroed
                uvll::set_data_for_uv_handle(watcher.native_handle(), null::<()>());
            }
        }
    }

    // NB: This is doing some sneaky things with borrowed pointers
    fn callback_from_watcher<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &W) -> &CB {
        unsafe {
            let handle = watcher.native_handle();
            let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
            assert handle_data.is_not_null();
            let cb = transmute::<&*c_void, &~CB>(&handle_data);
            return &**cb;
        }
    }
}
