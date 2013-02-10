// compile-flags: --test

// FIXME #4830 uncomment `priv` field modifiers

extern mod std;

use core::io::println;
use core::private::run_in_bare_thread;
use core::cast::{transmute, transmute_mut_region};
use core::libc::{uintptr_t, c_void};
use core::ptr::mut_null;
use context::Context;
use io::{EventLoop, IoFactory, Stream, TcpListener};
use uv_io::UvEventLoop;
use uv::ip4addr;

pub type EventLoopObject = uv_io::UvEventLoop;
pub type IoFactoryObject = uv_io::UvIoFactory;
pub type StreamObject = uv_io::UvStream;
pub type TcpListenerObject = uv_io::UvTcpListener;

fn macros() {
    macro_rules! rtdebug_ (
        ($( $arg:expr),+) => ( { } )
    )

    macro_rules! rtdebug (
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
    /*priv*/ event_loop: ~EventLoopObject,
    /// The scheduler's saved context.
    /// Always valid when a task is executing, otherwise not
    /*priv*/ saved_context: Context,
    /// The currently executing task
    /*priv*/ current_task: Option<~Task>,
    /// A queue of jobs to perform immediately upon return from task
    /// context to scheduler context.
    /// FIXME: This should probably be 'right after a context switch',
    /// and there is only ever one, call it CleanupAction, or PostContextSwitchAction
    /*priv*/ cleanup_jobs: ~[CleanupJob]
}

enum CleanupJob {
    RescheduleTask(~Task),
    RecycleTask(~Task),
    GiveTask(~Task, &fn(~Task))
}

impl Scheduler {

    static fn new(event_loop: ~EventLoopObject) -> Scheduler {
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
        // The running task should have passed ownership elsewhere
        assert self.current_task.is_none();

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

    /// Block a running task, context switch to the scheduler, then pass the
    /// blocked task to a closure.
    ///
    /// # Safety note
    ///
    /// The closure here is a *stack* closure that lives in the running task.
    /// It gets transmuted to the scheduler's lifetime and called while the task
    /// is blocked.
    fn block_running_task_and_then(&mut self, f: &fn(~Task)) {
        assert self.in_task_context();

        rtdebug!("blocking task");

        let blocked_task = self.current_task.swap_unwrap();
        let f_fake_region = unsafe { transmute::<&fn(~Task), &fn(~Task)>(f) };
        self.enqueue_cleanup_job(GiveTask(blocked_task, f_fake_region));
        let give_job = self.last_cleanup_job();
        let blocked_task: &mut Task = match give_job {
            &GiveTask(~ref mut task, _) => task, _ => fail!()
        };

        self.swap_out_task(blocked_task);
    }

    /// Switch directly to another task, without going through the scheduler.
    /// You would want to think hard about doing this, e.g. if there are
    /// pending I/O events it would be a bad idea.
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

    fn enqueue_cleanup_job(&mut self, job: CleanupJob/&self) {
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
                GiveTask(task, f) => f(task)
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

        let mut sched = ~UvEventLoop::new_scheduler();
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

        let mut sched = ~UvEventLoop::new_scheduler();
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

        let mut sched = ~UvEventLoop::new_scheduler();
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

        let mut sched = ~UvEventLoop::new_scheduler();

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

        let mut sched = ~UvEventLoop::new_scheduler();

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

#[test]
fn test_block_task() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            do use_thread_local_scheduler |sched| {
                assert sched.in_task_context();
                do sched.block_running_task_and_then() |task| {
                    assert !sched.in_task_context();
                    sched.task_queue.push_back(task);
                }
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}

#[test]
fn test_simple_io_no_connect() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let task = ~do Task::new(&mut sched.stack_pool) {
            do use_thread_local_scheduler |sched| {
                let io = sched.event_loop.io().unwrap();
                let addr = ip4addr("127.0.0.1", 2926);
                let maybe_chan = io.connect(addr);
                assert maybe_chan.is_none();
            }
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}

#[test]
fn test_simple_tcp_server_and_client() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let addr = ip4addr("127.0.0.1", 2929);

        let client_task = ~do Task::new(&mut sched.stack_pool) {
            do use_thread_local_scheduler |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut stream = io.connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.close();
            }
        };

        let server_task = ~do Task::new(&mut sched.stack_pool) {
            do use_thread_local_scheduler |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut listener = io.bind(addr).unwrap();
                let mut stream = listener.listen().unwrap();
                let mut buf = [0, .. 2048];
                let nread = stream.read(buf).unwrap();
                assert nread == 8;
                for uint::range(0, nread) |i| {
                    rtdebug!("%u", buf[i] as uint);
                    assert buf[i] == i as u8;
                }
                stream.close();
                listener.close();
            }
        };

        // Start the server first so it listens before the client connects
        sched.task_queue.push_back(server_task);
        sched.task_queue.push_back(client_task);
        sched.run();
    }
}

#[test] #[ignore]
fn test_read_buffering() {
    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let addr = ip4addr("127.0.0.1", 2930);

        let client_task = ~do Task::new(&mut sched.stack_pool) {
            do use_thread_local_scheduler |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut stream = io.connect(addr).unwrap();
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.write([0, 1, 2, 3, 4, 5, 6, 7]);
                stream.close();
            }
        };

        let server_task = ~do Task::new(&mut sched.stack_pool) {
            do use_thread_local_scheduler |sched| {
                let io = sched.event_loop.io().unwrap();
                let mut listener = io.bind(addr).unwrap();
                let mut stream = listener.listen().unwrap();
                let mut buf = [0, .. 2048];

                let expected = 32;
                let mut current = 0;
                let mut reads = 0;

                while current < expected {
                    let nread = stream.read(buf).unwrap();
                    for uint::range(0, nread) |i| {
                        let val = buf[i] as uint;
                        assert val == current % 8;
                        current += 1;
                    }
                    reads += 1;

                    do use_thread_local_scheduler |scheduler| {
                        // Yield to the other task in hopes that it will trigger
                        // a read callback while we are blocked.
                        do scheduler.block_running_task_and_then |task| {
                            scheduler.task_queue.push_back(task);
                        }
                    }
                }

                // Make sure we had multiple reads
                assert reads > 1;

                stream.close();
                listener.close();
            }
        };

        // Start the server first so it listens before the client connects
        sched.task_queue.push_back(server_task);
        sched.task_queue.push_back(client_task);
        sched.run();
    }
}

mod io {
    use std::net::ip::IpAddr;
    use super::{IoFactoryObject, StreamObject, TcpListenerObject};

    pub trait EventLoop {
        fn run(&mut self);
        fn callback(&mut self, ~fn());
        /// The asynchronous I/O services. Not all event loops may provide one
        /// NOTE: Should be IoFactory
        fn io(&mut self) -> Option<&self/mut IoFactoryObject>;
    }

    pub trait IoFactory {
        fn connect(&mut self, addr: IpAddr) -> Option<~StreamObject>;
        fn bind(&mut self, addr: IpAddr) -> Option<~TcpListenerObject>;
    }

    pub trait TcpListener {
        fn listen(&mut self) -> Option<~StreamObject>;
    }

    pub trait Stream {
        fn read(&mut self, buf: &mut [u8]) -> Result<uint, ()>;
        fn write(&mut self, buf: &[u8]) -> Result<(), ()>;
    }
}

mod uv_io {

    use super::uv::*;
    use super::io::*;
    use std::net::ip::IpAddr;
    use std::cell::{Cell, empty_cell};
    use super::StreamObject;
    use super::use_thread_local_scheduler;
    use super::Scheduler;
    use super::uv::*;
    use super::io::IoFactory;
    use super::IoFactoryObject;

    pub struct UvEventLoop {
        uvio: UvIoFactory
    }

    pub impl UvEventLoop {
        static fn new() -> UvEventLoop {
            UvEventLoop {
                uvio: UvIoFactory(Loop::new())
            }
        }

        /// A convenience constructor
        static fn new_scheduler() -> Scheduler {
            Scheduler::new(~UvEventLoop::new())
        }
    }

    pub impl UvEventLoop: Drop {
        fn finalize(&self) {
            // XXX: Need mutable finalizer
            let self = unsafe { transmute::<&UvEventLoop, &mut UvEventLoop>(self) };
            let mut uv_loop = self.uvio.uv_loop();
            uv_loop.close();
        }
    }

    pub impl UvEventLoop: EventLoop {

        fn run(&mut self) {
            self.uvio.uv_loop().run();
        }

        fn callback(&mut self, f: ~fn()) {
            let mut idle_watcher =  IdleWatcher::new(self.uvio.uv_loop());
            do idle_watcher.start |idle_watcher, status| {
                assert status.is_none();
                let mut idle_watcher = idle_watcher;
                idle_watcher.stop();
                idle_watcher.close();
                f();
            }
        }

        fn io(&mut self) -> Option<&self/mut IoFactoryObject> {
            Some(&mut self.uvio)
        }
    }

    #[test]
    fn test_callback_run_once() {
        do run_in_bare_thread {
            let mut event_loop = UvEventLoop::new();
            let mut count = 0;
            let count_ptr: *mut int = &mut count;
            do event_loop.callback {
                unsafe { *count_ptr += 1 }
            }
            event_loop.run();
            assert count == 1;
        }
    }

    pub struct UvIoFactory(Loop);

    pub impl UvIoFactory {
        fn uv_loop(&mut self) -> &self/mut Loop {
            match self { &UvIoFactory(ref mut ptr) => ptr }
        }
    }

    pub impl UvIoFactory: IoFactory {
        // Connect to an address and return a new stream
        // NB: This blocks the task waiting on the connection.
        // It would probably be better to return a future
        fn connect(&mut self, addr: IpAddr) -> Option<~StreamObject> {
            // Create a cell in the task to hold the result. We will fill
            // the cell before resuming the task.
            let result_cell = empty_cell();
            let result_cell_ptr: *Cell<Option<~StreamObject>> = &result_cell;

            do use_thread_local_scheduler |scheduler| {
                assert scheduler.in_task_context();

                // Block this task and take ownership, switch to scheduler context
                do scheduler.block_running_task_and_then |task| {

                    rtdebug!("connect: entered scheduler context");
                    assert !scheduler.in_task_context();
                    let mut tcp_watcher = TcpWatcher::new(self.uv_loop());
                    let task_cell = Cell(task);

                    // Wait for a connection
                    do tcp_watcher.connect(addr) |stream_watcher, status| {
                        rtdebug!("connect: in connect callback");
                        let maybe_stream = if status.is_none() {
                            rtdebug!("status is none");
                            Some(~UvStream(stream_watcher))
                        } else {
                            rtdebug!("status is some");
                            stream_watcher.close(||());
                            None
                        };

                        // Store the stream in the task's stack
                        unsafe { (*result_cell_ptr).put_back(maybe_stream); }

                        // Context switch
                        do use_thread_local_scheduler |scheduler| {
                            scheduler.resume_task_immediately(task_cell.take());
                        }
                    }
                }
            }

            assert !result_cell.is_empty();
            return result_cell.take();
        }

        fn bind(&mut self, addr: IpAddr) -> Option<~TcpListenerObject> {
            let mut watcher = TcpWatcher::new(self.uv_loop());
            watcher.bind(addr);
            return Some(~UvTcpListener(watcher));
        }
    }

    pub struct UvTcpListener(TcpWatcher);

    impl UvTcpListener {
        fn watcher(&self) -> TcpWatcher {
            match self { &UvTcpListener(w) => w }
        }

        fn close(&self) {
            // FIXME: Need to wait until close finishes before returning
            self.watcher().as_stream().close(||());
        }
    }

    impl UvTcpListener: Drop {
        fn finalize(&self) {
            // FIXME: Again, this never gets called. Use .close() instead
            //self.watcher().as_stream().close(||());
        }
    }

    impl UvTcpListener: TcpListener {

        fn listen(&mut self) -> Option<~StreamObject> {
            rtdebug!("entering listen");
            let result_cell = empty_cell();
            let result_cell_ptr: *Cell<Option<~StreamObject>> = &result_cell;

            let server_tcp_watcher = self.watcher();

            do use_thread_local_scheduler |scheduler| {
                assert scheduler.in_task_context();

                do scheduler.block_running_task_and_then |task| {
                    let task_cell = Cell(task);
                    let mut server_tcp_watcher = server_tcp_watcher;
                    do server_tcp_watcher.listen |server_stream_watcher, status| {
                        let maybe_stream = if status.is_none() {
                            let mut server_stream_watcher = server_stream_watcher;
                            let mut loop_ = loop_from_watcher(&server_stream_watcher);
                            let mut client_tcp_watcher = TcpWatcher::new(&mut loop_);
                            let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                            // FIXME: Need's to be surfaced in interface
                            server_stream_watcher.accept(client_tcp_watcher);
                            Some(~UvStream::new(client_tcp_watcher))
                        } else {
                            None
                        };

                        unsafe { (*result_cell_ptr).put_back(maybe_stream); }

                        rtdebug!("resuming task from listen");
                        // Context switch
                        do use_thread_local_scheduler |scheduler| {
                            scheduler.resume_task_immediately(task_cell.take());
                        }
                    }
                }
            }

            assert !result_cell.is_empty();
            return result_cell.take();
        }
    }

    pub struct UvStream(StreamWatcher);

    impl UvStream {
        static fn new(watcher: StreamWatcher) -> UvStream {
            UvStream(watcher)
        }

        fn watcher(&self) -> StreamWatcher {
            match self { &UvStream(w) => w }
        }

        // FIXME: finalize isn't working for ~UvStream???
        fn close(&self) {
            // FIXME: Need to wait until this finishes before returning
            self.watcher().close(||());
        }
    }

    impl UvStream: Drop {
        fn finalize(&self) {
            rtdebug!("closing stream");
            //self.watcher().close(||());
        }
    }

    pub impl UvStream: Stream {
        fn read(&mut self, buf: &mut [u8]) -> Result<uint, ()> {
            let result_cell = empty_cell();
            let result_cell_ptr: *Cell<Result<uint, ()>> = &result_cell;

            do use_thread_local_scheduler |scheduler| {
                assert scheduler.in_task_context();
                let watcher = self.watcher();
                let buf_ptr: *&mut [u8] = &buf;
                do scheduler.block_running_task_and_then |task| {
                    rtdebug!("read: entered scheduler context");
                    assert !scheduler.in_task_context();
                    let mut watcher = watcher;
                    let task_cell = Cell(task);
                    let alloc: AllocCallback = |_| unsafe { slice_to_uv_buf(*buf_ptr) };
                    do watcher.read_start(alloc) |_watcher, nread, _buf, status| {
                        let result = if status.is_none() {
                            assert nread >= 0;
                            Ok(nread as uint)
                        } else {
                            Err(())
                        };

                        unsafe { (*result_cell_ptr).put_back(result); }

                        do use_thread_local_scheduler |scheduler| {
                            scheduler.resume_task_immediately(task_cell.take());
                        }
                    }
                }
            }

            assert !result_cell.is_empty();
            return result_cell.take();
        }

        fn write(&mut self, buf: &[u8]) -> Result<(), ()> {
            let result_cell = empty_cell();
            let result_cell_ptr: *Cell<Result<(), ()>> = &result_cell;
            do use_thread_local_scheduler |scheduler| {
                assert scheduler.in_task_context();
                let watcher = self.watcher();
                let buf_ptr: *&[u8] = &buf;
                do scheduler.block_running_task_and_then |task| {
                    let mut watcher = watcher;
                    let task_cell = Cell(task);
                    let buf = unsafe { &*buf_ptr };
                    // FIXME: OMGCOPIES
                    let buf = buf.to_vec();
                    do watcher.write(buf) |_watcher, status| {
                        let result = if status.is_none() {
                            Ok(())
                        } else {
                            Err(())
                        };

                        unsafe { (*result_cell_ptr).put_back(result); }

                        do use_thread_local_scheduler |scheduler| {
                            scheduler.resume_task_immediately(task_cell.take());
                        }
                    }
                }
            }

            assert !result_cell.is_empty();
            return result_cell.take();
        }
    }
}

pub mod uv {
    use core::private::run_in_bare_thread;
    use core::str::raw::from_c_str;
    use core::libc::{c_void, c_int, size_t, malloc, free, ssize_t};
    use core::cast::transmute;
    use core::ptr::null;
    use core::sys::size_of;
    use std::net::ip::*;
    use std::cell::Cell;
    use uvll = std::uv::ll;
    use super::Thread;

    type Buf = uvll::uv_buf_t;

    // A little helper. Can't figure out how to do this with net::ip
    pub fn ip4addr(addr: &str, port: uint) -> IpAddr {
        Ipv4(unsafe { uvll::ip4_addr(addr, port as int) })
    }

    /// A trait for callbacks to implement. Provides a little extra type safety
    /// for generic interop functions like `set_watcher_callback`.
    trait Callback { }

    type NullCallback = ~fn();
    impl NullCallback: Callback { }

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
        static pub fn from_native_handle(T) -> Self;
        pub fn native_handle(&self) -> T;
    }

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
            NativeHandle::from_native_handle(handle)
        }

        fn run(&mut self) {
            unsafe { uvll::run(self.native_handle()) };
        }

        fn close(&mut self) {
            unsafe { uvll::loop_delete(self.native_handle()) };
        }
    }

    pub impl Loop: NativeHandle<*uvll::uv_loop_t> {
        static fn from_native_handle(handle: *uvll::uv_loop_t) -> Loop {
            Loop { handle: handle }
        }
        fn native_handle(&self) -> *uvll::uv_loop_t {
            self.handle
        }
    }

    #[test]
    fn loop_smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            loop_.run();
            loop_.close();
        }
    }

    /// The trait implemented by uv 'watchers' (handles). Watchers
    /// are non-owning wrappers around the uv handles and are not completely safe -
    /// there may be multiple instances for a single underlying handle.
    /// Watchers are generally created, then `start`ed, `stop`ed and `close`ed,
    /// but due to their complex life cycle may not be entirely memory safe
    /// if used in unanticipated patterns.
    trait Watcher {
        fn event_loop(&self) -> Loop;
    }

    pub struct IdleWatcher(*uvll::uv_idle_t);

    impl IdleWatcher: Watcher {
        fn event_loop(&self) -> Loop {
            loop_from_watcher(self)
        }
    }

    type IdleCallback = ~fn(IdleWatcher, Option<UvError>);
    impl IdleCallback: Callback { }

    pub impl IdleWatcher {
        static fn new(loop_: &mut Loop) -> IdleWatcher {
            unsafe {
                let handle = uvll::idle_new();
                assert handle.is_not_null();
                assert 0 == uvll::idle_init(loop_.native_handle(), handle);
                uvll::set_data_for_uv_handle(handle, null::<()>());
                NativeHandle::from_native_handle(handle)
            }
        }

        fn start(&mut self, cb: IdleCallback) {

            set_watcher_callback(self, cb);
            unsafe { assert 0 == uvll::idle_start(self.native_handle(), idle_cb) };

            extern fn idle_cb(handle: *uvll::uv_idle_t, status: c_int) {
                let idle_watcher: IdleWatcher = NativeHandle::from_native_handle(handle);
                let cb: &IdleCallback = borrow_callback_from_watcher(&idle_watcher);
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
                let mut idle_watcher = NativeHandle::from_native_handle(handle);
                drop_watcher_callback::<uvll::uv_idle_t, IdleWatcher, IdleCallback>(&mut idle_watcher);
                unsafe { uvll::idle_delete(handle) };
            }
        }
    }

    pub impl IdleWatcher: NativeHandle<*uvll::uv_idle_t> {
        static fn from_native_handle(handle: *uvll::uv_idle_t) -> IdleWatcher {
            IdleWatcher(handle)
        }
        fn native_handle(&self) -> *uvll::uv_idle_t {
            match self { &IdleWatcher(ptr) => ptr }
        }
    }

    // uv_stream t is the parent class of uv_tcp_t, uv_pipe_t, uv_tty_t and uv_file_t
    pub struct StreamWatcher(*uvll::uv_stream_t);

    impl StreamWatcher: Watcher {
        fn event_loop(&self) -> Loop {
            loop_from_watcher(self)
        }
    }

    type ReadCallback = ~fn(StreamWatcher, int, Buf, Option<UvError>);
    impl ReadCallback: Callback { }

    // XXX: The uv alloc callback also has a *uv_handle_t arg
    type AllocCallback = ~fn(uint) -> Buf;
    impl AllocCallback: Callback { }

    pub impl StreamWatcher {

        fn read_start(&mut self, alloc: AllocCallback, cb: ReadCallback) {
            let data = get_watcher_data(self);
            assert data.alloc_cb.is_none();
            data.alloc_cb = Some(alloc);
            assert data.read_cb.is_none();
            data.read_cb = Some(cb);

            let handle = self.native_handle();
            unsafe { uvll::read_start(handle, alloc_cb, read_cb) };

            extern fn alloc_cb(stream: *uvll::uv_stream_t, suggested_size: size_t) -> Buf {
                let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
                let data = get_watcher_data(&mut stream_watcher);
                let alloc_cb = data.alloc_cb.get_ref();
                return (*alloc_cb)(suggested_size as uint);
            }

            extern fn read_cb(stream: *uvll::uv_stream_t, nread: ssize_t, ++buf: Buf) {
                rtdebug!("read_cb");
                rtdebug!("buf addr: %x", buf.base as uint);
                rtdebug!("buf len: %d", buf.len as int);
                let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(stream);
                let data = get_watcher_data(&mut stream_watcher);
                let cb = data.read_cb.get_ref();
                let status = status_to_maybe_uv_error(stream, nread as c_int);
                (*cb)(stream_watcher, nread as int, buf, status);
            }
        }

        fn write(&mut self, msg: ~[u8], cb: ConnectionCallback) {
            let data = get_watcher_data(self);
            assert data.write_cb.is_none();
            data.write_cb = Some(cb);

            let req = WriteRequest::new();
            let buf = vec_to_uv_buf(msg);
            // FIXME: Allocation
            let bufs = ~[buf];
            unsafe { assert 0 == uvll::write(req.native_handle(),
                                             self.native_handle(),
                                             &bufs, write_cb); }
            // XXX: Freeing immediately after write. Is this ok?
            let _v = vec_from_uv_buf(buf);

            extern fn write_cb(req: *uvll::uv_write_t, status: c_int) {
                let write_request: WriteRequest = NativeHandle::from_native_handle(req);
                let mut stream_watcher = write_request.stream();
                write_request.delete();
                let cb = get_watcher_data(&mut stream_watcher).write_cb.swap_unwrap();
                let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
                cb(stream_watcher, status);
            }
        }

        fn accept(&mut self, stream: StreamWatcher) {
            let self_handle = self.native_handle() as *c_void;
            let stream_handle = stream.native_handle() as *c_void;
            unsafe { assert 0 == uvll::accept(self_handle, stream_handle); }
        }

        fn close(self, cb: NullCallback) {
            {
                let mut self = self;
                let data = get_watcher_data(&mut self);
                assert data.close_cb.is_none();
                data.close_cb = Some(cb);
            }

            unsafe { uvll::close(self.native_handle(), close_cb); }

            extern fn close_cb(handle: *uvll::uv_stream_t) {
                let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
                {
                    let mut data = get_watcher_data(&mut stream_watcher);
                    data.close_cb.swap_unwrap()();
                }
                drop_watcher_data(&mut stream_watcher);
                unsafe { free(handle as *c_void) }
            }
        }
    }

    pub impl StreamWatcher: NativeHandle<*uvll::uv_stream_t> {
        static fn from_native_handle(handle: *uvll::uv_stream_t) -> StreamWatcher {
            StreamWatcher(handle)
        }
        fn native_handle(&self) -> *uvll::uv_stream_t {
            match self { &StreamWatcher(ptr) => ptr }
        }
    }

    pub struct TcpWatcher(*uvll::uv_tcp_t);

    impl TcpWatcher: Watcher {
        fn event_loop(&self) -> Loop {
            loop_from_watcher(self)
        }
    }

    type ConnectionCallback = ~fn(StreamWatcher, Option<UvError>);
    impl ConnectionCallback: Callback { }

    pub impl TcpWatcher {
        static fn new(loop_: &mut Loop) -> TcpWatcher {
            unsafe {
                let handle = malloc(size_of::<uvll::uv_tcp_t>() as size_t) as *uvll::uv_tcp_t;
                assert handle.is_not_null();
                assert 0 == uvll::tcp_init(loop_.native_handle(), handle);
                let mut watcher = NativeHandle::from_native_handle(handle);
                install_watcher_data(&mut watcher);
                return watcher;
            }
        }

        fn bind(&mut self, address: IpAddr) {
            match address {
                Ipv4(addr) => {
                    let result = unsafe { uvll::tcp_bind(self.native_handle(), &addr) };
                    // FIXME: bind is likely to fail. need real error handling
                    assert result == 0;
                }
                _ => fail!()
            }
        }

        fn connect(&mut self, address: IpAddr, cb: ConnectionCallback) {
            unsafe {
                assert get_watcher_data(self).connect_cb.is_none();
                get_watcher_data(self).connect_cb = Some(cb);

                let mut connect_watcher = ConnectRequest::new();
                let connect_handle = connect_watcher.native_handle();
                match address {
                    Ipv4(addr) => {
                        rtdebug!("connect_t: %x", connect_handle as uint);
                        assert 0 == uvll::tcp_connect(connect_handle, self.native_handle(), &addr, connect_cb);
                    }
                    _ => fail!()
                }

                extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
                    rtdebug!("connect_t: %x", req as uint);
                    let connect_request: ConnectRequest = NativeHandle::from_native_handle(req);
                    let mut stream_watcher = connect_request.stream();
                    connect_request.delete();
                    let cb: ConnectionCallback = get_watcher_data(&mut stream_watcher).connect_cb.swap_unwrap();
                    let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
                    cb(stream_watcher, status);
                }
            }
        }

        fn listen(&mut self, cb: ConnectionCallback) {
            let data = get_watcher_data(self);
            assert data.connect_cb.is_none();
            data.connect_cb = Some(cb);

            unsafe {
                const BACKLOG: c_int = 128; // FIXME
                // FIXME: This can probably fail
                assert 0 == uvll::listen(self.native_handle(), BACKLOG, connection_cb);
            }

            extern fn connection_cb(handle: *uvll::uv_stream_t, status: c_int) {
                rtdebug!("connection_cb");
                let mut stream_watcher: StreamWatcher = NativeHandle::from_native_handle(handle);
                let cb = get_watcher_data(&mut stream_watcher).connect_cb.swap_unwrap();
                let status = status_to_maybe_uv_error(stream_watcher.native_handle(), status);
                cb(stream_watcher, status);
            }
        }

        fn as_stream(&self) -> StreamWatcher {
            NativeHandle::from_native_handle(self.native_handle() as *uvll::uv_stream_t)
        }
    }

    pub impl TcpWatcher: NativeHandle<*uvll::uv_tcp_t> {
        static fn from_native_handle(handle: *uvll::uv_tcp_t) -> TcpWatcher {
            TcpWatcher(handle)
        }
        fn native_handle(&self) -> *uvll::uv_tcp_t {
            match self { &TcpWatcher(ptr) => ptr }
        }
    }

    trait Request { }

    type ConnectCallback = ~fn(ConnectRequest, Option<UvError>);
    impl ConnectCallback: Callback { }

    // uv_connect_t is a subclass of uv_req_t
    struct ConnectRequest(*uvll::uv_connect_t);

    impl ConnectRequest: Request { }

    impl ConnectRequest {

        static fn new() -> ConnectRequest {
            let connect_handle = unsafe { malloc(size_of::<uvll::uv_connect_t>() as size_t) };
            assert connect_handle.is_not_null();
            let connect_handle = connect_handle as *uvll::uv_connect_t;
            ConnectRequest(connect_handle)
        }

        fn stream(&self) -> StreamWatcher {
            unsafe {
                let stream_handle = uvll::get_stream_handle_from_connect_req(self.native_handle());
                NativeHandle::from_native_handle(stream_handle)
            }
        }

        fn delete(self) {
            unsafe { free(self.native_handle() as *c_void) }
        }
    }

    pub impl ConnectRequest: NativeHandle<*uvll::uv_connect_t> {
        static fn from_native_handle(handle: *uvll:: uv_connect_t) -> ConnectRequest {
            ConnectRequest(handle)
        }
        fn native_handle(&self) -> *uvll::uv_connect_t {
            match self { &ConnectRequest(ptr) => ptr }
        }
    }

    pub struct WriteRequest(*uvll::uv_write_t);

    impl WriteRequest: Request { }

    impl WriteRequest {

        static fn new() -> WriteRequest {
            let write_handle = unsafe { malloc(size_of::<uvll::uv_write_t>() as size_t) };
            assert write_handle.is_not_null();
            let write_handle = write_handle as *uvll::uv_write_t;
            WriteRequest(write_handle)
        }

        fn stream(&self) -> StreamWatcher {
            unsafe {
                let stream_handle = uvll::get_stream_handle_from_write_req(self.native_handle());
                NativeHandle::from_native_handle(stream_handle)
            }
        }

        fn delete(self) {
            unsafe { free(self.native_handle() as *c_void) }
        }
    }

    pub impl WriteRequest: NativeHandle<*uvll::uv_write_t> {
        static fn from_native_handle(handle: *uvll:: uv_write_t) -> WriteRequest {
            WriteRequest(handle)
        }
        fn native_handle(&self) -> *uvll::uv_write_t {
            match self { &WriteRequest(ptr) => ptr }
        }
    }

    // FIXME: Follow the pattern below by parameterizing over T: Watcher, not T
    fn status_to_maybe_uv_error<T>(handle: *T, status: c_int) -> Option<UvError> {
        if status != -1 {
            None
        } else {
            unsafe {
                rtdebug!("handle: %x", handle as uint);
                let loop_ = uvll::get_loop_for_uv_handle(handle);
                rtdebug!("loop: %x", loop_ as uint);
                let err = uvll::last_error(loop_);
                Some(UvError(err))
            }
        }
    }

    fn loop_from_watcher<H, W: Watcher NativeHandle<*H>>(watcher: &W) -> Loop {
        let handle = watcher.native_handle();
        let loop_ = unsafe { uvll::get_loop_for_uv_handle(handle) };
        NativeHandle::from_native_handle(loop_)
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
    fn borrow_callback_from_watcher<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &W) -> &CB {
        unsafe {
            let handle = watcher.native_handle();
            let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
            assert handle_data.is_not_null();
            let cb = transmute::<&*c_void, &~CB>(&handle_data);
            return &**cb;
        }
    }

    fn take_callback_from_watcher<H, W: Watcher NativeHandle<*H>, CB: Callback>(watcher: &mut W) -> CB {
        unsafe {
            let handle = watcher.native_handle();
            let handle_data: *c_void = uvll::get_data_for_uv_handle(handle);
            assert handle_data.is_not_null();
            uvll::set_data_for_uv_handle(handle, null::<()>());
            let cb: ~CB = transmute::<*c_void, ~CB>(handle_data);
            let cb = match cb { ~cb => cb };
            return cb;
        }
    }

    struct WatcherData {
        read_cb: Option<ReadCallback>,
        write_cb: Option<ConnectionCallback>,
        connect_cb: Option<ConnectionCallback>,
        close_cb: Option<NullCallback>,
        alloc_cb: Option<AllocCallback>
    }

    fn install_watcher_data<H, W: Watcher NativeHandle<*H>>(watcher: &mut W) {
        unsafe {
            let data = ~WatcherData {
                read_cb: None,
                write_cb: None,
                connect_cb: None,
                close_cb: None,
                alloc_cb: None
            };
            let data = transmute::<~WatcherData, *c_void>(data);
            uvll::set_data_for_uv_handle(watcher.native_handle(), data);
        }
    }

    fn get_watcher_data<H, W: Watcher NativeHandle<*H>>(watcher: &r/mut W) -> &r/mut WatcherData {
        unsafe {
            let data = uvll::get_data_for_uv_handle(watcher.native_handle());
            let data = transmute::<&*c_void, &mut ~WatcherData>(&data);
            return &mut **data;
        }
    }
    
    fn drop_watcher_data<H, W: Watcher NativeHandle<*H>>(watcher: &mut W) {
        unsafe {
            let data = uvll::get_data_for_uv_handle(watcher.native_handle());
            let _data = transmute::<*c_void, ~WatcherData>(data);
            uvll::set_data_for_uv_handle(watcher.native_handle(), null::<()>());
        }
    }

    #[test]
    fn test_slice_to_uv_buf() {
        let slice = [0, .. 20];
        let buf = slice_to_uv_buf(slice);

        assert buf.len == 20;

        unsafe {
            let base = transmute::<*u8, *mut u8>(buf.base);
            (*base) = 1;
            (*ptr::mut_offset(base, 1)) = 2;
        }

        assert slice[0] == 1;
        assert slice[1] == 2;
    }

    fn slice_to_uv_buf(v: &[u8]) -> Buf {
        let data = unsafe { vec::raw::to_ptr(v) };
        unsafe { uvll::buf_init(data, v.len()) }
    }

    // FIXME: Do these conversions without copying
    fn vec_to_uv_buf(v: ~[u8]) -> Buf {
        let data = unsafe { malloc(v.len() as size_t) } as *u8;
        assert data.is_not_null();
        do vec::as_imm_buf(v) |b, l| {
            let data = data as *mut u8;
            unsafe { ptr::copy_memory(data, b, l) }
        }
        let buf = unsafe { uvll::buf_init(data, v.len()) };
        return buf;
    }

    fn vec_from_uv_buf(buf: Buf) -> Option<~[u8]> {
        if !(buf.len == 0 && buf.base.is_null()) {
            let v = unsafe { vec::from_buf(buf.base, buf.len as uint) };
            unsafe { free(buf.base as *c_void) };
            return Some(v);
        } else {
            // No buffer
            return None;
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
            loop_.close();
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
                    assert status.is_none();
                    let mut idle_watcher = idle_watcher;
                    idle_watcher.stop();
                    idle_watcher.close();
                }
            }
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn connect_close() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            // Connect to a port where nobody is listening
            let addr = ip4addr("127.0.0.1", 2923);
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                rtdebug!("tcp_watcher.connect!");
                assert status.is_some();
                assert status.get().name() == ~"ECONNREFUSED";
                stream_watcher.close(||());
            }
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    #[ignore(reason = "need a server to connect to")]
    fn connect_read() {
        do run_in_bare_thread() {
            let mut loop_ = Loop::new();
            let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
            let addr = ip4addr("127.0.0.1", 2924);
            do tcp_watcher.connect(addr) |stream_watcher, status| {
                let mut stream_watcher = stream_watcher;
                rtdebug!("tcp_watcher.connect!");
                assert status.is_none();
                let alloc: AllocCallback = |size| vec_to_uv_buf(vec::from_elem(size, 0));
                do stream_watcher.read_start(alloc) |stream_watcher, nread, buf, status| {
                    let buf = vec_from_uv_buf(buf);
                    rtdebug!("read cb!");
                    if status.is_none() {
                        let bytes = buf.unwrap();
                        rtdebug!("%s", bytes.slice(0, nread as uint).to_str());
                    } else {
                        rtdebug!("status after read: %s", status.get().to_str());
                        rtdebug!("closing");
                        stream_watcher.close(||());
                    }
                }
            }
            loop_.run();
            loop_.close();
        }
    }

    #[test]
    fn listen() {
        do run_in_bare_thread() {
            const MAX: int = 10;
            let mut loop_ = Loop::new();
            let mut server_tcp_watcher = { TcpWatcher::new(&mut loop_) };
            let addr = ip4addr("127.0.0.1", 2925);
            server_tcp_watcher.bind(addr);
            let loop_ = loop_;
            rtdebug!("listening");
            do server_tcp_watcher.listen |server_stream_watcher, status| {
                rtdebug!("listened!");
                assert status.is_none();
                let mut server_stream_watcher = server_stream_watcher;
                let mut loop_ = loop_;
                let mut client_tcp_watcher = TcpWatcher::new(&mut loop_);
                let mut client_tcp_watcher = client_tcp_watcher.as_stream();
                server_stream_watcher.accept(client_tcp_watcher);
                let count_cell = Cell(0);
                let server_stream_watcher = server_stream_watcher;
                rtdebug!("starting read");
                let alloc: AllocCallback = |size| vec_to_uv_buf(vec::from_elem(size, 0));
                do client_tcp_watcher.read_start(alloc) |stream_watcher, nread, buf, status| {
                    rtdebug!("i'm reading!");
                    let buf = vec_from_uv_buf(buf);
                    let mut count = count_cell.take();
                    if status.is_none() {
                        rtdebug!("got %d bytes", nread);
                        let buf = buf.unwrap();
                        for buf.view(0, nread as uint).each |byte| {
                            assert *byte == count as u8;
                            rtdebug!("%u", *byte as uint);
                            count += 1;
                        }
                    } else {
                        assert count == MAX;
                        do stream_watcher.close {
                            server_stream_watcher.close(||());
                        }
                    }
                    count_cell.put_back(count);
                }
            }

            let _client_thread = do Thread::start {
                rtdebug!("starting client thread");
                let mut loop_ = Loop::new();
                let mut tcp_watcher = { TcpWatcher::new(&mut loop_) };
                do tcp_watcher.connect(addr) |stream_watcher, status| {
                    rtdebug!("connecting");
                    assert status.is_none();
                    let mut stream_watcher = stream_watcher;
                    let msg = ~[0, 1, 2, 3, 4, 5, 6 ,7 ,8, 9];
                    do stream_watcher.write(msg) |stream_watcher, status| {
                        rtdebug!("writing");
                        assert status.is_none();
                        stream_watcher.close(||());
                    }
                }
                loop_.run();
                loop_.close();
            };

            let mut loop_ = loop_;
            loop_.run();
            loop_.close();
        }
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

pub fn use_thread_local_scheduler(f: &fn(&mut Scheduler)) {
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
    let scheduler = ~UvEventLoop::new_scheduler();
    let mut tls_scheduler = ThreadLocalScheduler::new();
    tls_scheduler.put_scheduler(scheduler);
    {
        let _scheduler = tls_scheduler.get_scheduler();
    }
    let _scheduler = tls_scheduler.take_scheduler();
}

#[test]
fn thread_local_scheduler_two_instances() {
    let scheduler = ~UvEventLoop::new_scheduler();
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

type raw_thread = c_void;

struct Thread {
    main: ~fn(),
    raw_thread: *raw_thread
}

impl Thread {
    static fn start(main: ~fn()) -> Thread {
        fn substart(main: &fn()) -> *raw_thread {
            unsafe { rust_raw_thread_start(main) }
        }
        let raw = substart(main);
        Thread {
            main: main,
            raw_thread: raw
        }
    }
}

impl Thread: Drop {
    fn finalize(&self) {
        unsafe { rust_raw_thread_join_delete(self.raw_thread) }
    }
}

extern {
    pub unsafe fn rust_raw_thread_start(f: &fn()) -> *raw_thread;
    pub unsafe fn rust_raw_thread_join_delete(thread: *raw_thread);
}
