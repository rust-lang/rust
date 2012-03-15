#include "rust_scheduler.h"
#include "rust_util.h"

rust_scheduler::rust_scheduler(rust_kernel *kernel,
			       rust_srv *srv,
			       size_t num_threads,
			       rust_sched_id id) :
    kernel(kernel),
    srv(srv),
    env(srv->env),
    live_threads(num_threads),
    live_tasks(0),
    num_threads(num_threads),
    id(id)
{
    isaac_init(kernel, &rctx);
    create_task_threads();
}

rust_scheduler::~rust_scheduler() {
    destroy_task_threads();
}

rust_task_thread *
rust_scheduler::create_task_thread(int id) {
    rust_srv *srv = this->srv->clone();
    rust_task_thread *thread =
        new (kernel, "rust_task_thread") rust_task_thread(this, srv, id);
    KLOG(kernel, kern, "created task thread: " PTR ", id: %d, index: %d",
          thread, id, thread->list_index);
    return thread;
}

void
rust_scheduler::destroy_task_thread(rust_task_thread *thread) {
    KLOG(kernel, kern, "deleting task thread: " PTR ", name: %s, index: %d",
        thread, thread->name, thread->list_index);
    rust_srv *srv = thread->srv;
    delete thread;
    delete srv;
}

void
rust_scheduler::create_task_threads() {
    KLOG(kernel, kern, "Using %d scheduler threads.", num_threads);

    for(size_t i = 0; i < num_threads; ++i) {
        threads.push(create_task_thread(i));
    }
}

void
rust_scheduler::destroy_task_threads() {
    for(size_t i = 0; i < num_threads; ++i) {
        destroy_task_thread(threads[i]);
    }
}

void
rust_scheduler::start_task_threads()
{
    for(size_t i = 0; i < num_threads; ++i) {
        rust_task_thread *thread = threads[i];
        thread->start();
    }
}

void
rust_scheduler::join_task_threads()
{
    for(size_t i = 0; i < num_threads; ++i) {
        rust_task_thread *thread = threads[i];
        thread->join();
    }
}

void
rust_scheduler::kill_all_tasks() {
    for(size_t i = 0; i < num_threads; ++i) {
        rust_task_thread *thread = threads[i];
        thread->kill_all_tasks();
    }
}

rust_task *
rust_scheduler::create_task(rust_task *spawner, const char *name,
			    size_t init_stack_sz) {
    size_t thread_no;
    {
	scoped_lock with(lock);
	thread_no = isaac_rand(&rctx) % num_threads;
	live_tasks++;
    }
    rust_task_thread *thread = threads[thread_no];
    return thread->create_task(spawner, name, init_stack_sz);
}

rust_task *
rust_scheduler::create_task(rust_task *spawner, const char *name) {
    return create_task(spawner, name, env->min_stack_size);
}

void
rust_scheduler::release_task() {
    bool need_exit = false;
    {
	scoped_lock with(lock);
	live_tasks--;
	if (live_tasks == 0) {
	    need_exit = true;
	}
    }
    if (need_exit) {
	// There are no more tasks on this scheduler. Time to leave
	exit();
    }
}

void
rust_scheduler::exit() {
    // Take a copy of num_threads. After the last thread exits this
    // scheduler will get destroyed, and our fields will cease to exist.
    size_t current_num_threads = num_threads;
    for(size_t i = 0; i < current_num_threads; ++i) {
        threads[i]->exit();
    }
}

size_t
rust_scheduler::number_of_threads() {
    return num_threads;
}

void
rust_scheduler::release_task_thread() {
    I(this, !lock.lock_held_by_current_thread());
    uintptr_t new_live_threads;
    {
	scoped_lock with(lock);
	new_live_threads = --live_threads;
    }
    if (new_live_threads == 0) {
	kernel->release_scheduler_id(id);
    }
}
