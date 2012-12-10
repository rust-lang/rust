// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#ifndef RUST_SCHED_LAUNCHER_H
#define RUST_SCHED_LAUNCHER_H

#include "sync/rust_thread.h"
#include "rust_sched_driver.h"
#include "rust_kernel.h"
#include "rust_sched_loop.h"

class rust_sched_launcher : public kernel_owned<rust_sched_launcher> {
public:
    rust_kernel *kernel;

private:
    rust_sched_loop sched_loop;

protected:
    rust_sched_driver driver;

public:
    rust_sched_launcher(rust_scheduler *sched, int id, bool killed);
    virtual ~rust_sched_launcher() { }

    virtual void start() = 0;
    virtual void join() = 0;
    rust_sched_loop *get_loop() { return &sched_loop; }
};

class rust_thread_sched_launcher
  :public rust_sched_launcher,
   private rust_thread {
public:
    rust_thread_sched_launcher(rust_scheduler *sched, int id, bool killed);
    virtual void start() { rust_thread::start(); }
    virtual void join() { rust_thread::join(); }
    virtual void run() { driver.start_main_loop(); }
};

class rust_manual_sched_launcher : public rust_sched_launcher {
public:
    rust_manual_sched_launcher(rust_scheduler *sched, int id, bool killed);
    virtual void start() { }
    virtual void join() { }
    rust_sched_driver *get_driver() { return &driver; };
};

class rust_sched_launcher_factory {
public:
    virtual ~rust_sched_launcher_factory() { }
    virtual rust_sched_launcher *
    create(rust_scheduler *sched, int id, bool killed) = 0;
};

class rust_thread_sched_launcher_factory
    : public rust_sched_launcher_factory {
public:
    virtual rust_sched_launcher *create(rust_scheduler *sched, int id,
                                        bool killed);
};

class rust_manual_sched_launcher_factory
    : public rust_sched_launcher_factory {
private:
    rust_manual_sched_launcher *launcher;
public:
    rust_manual_sched_launcher_factory() : launcher(NULL) { }
    virtual rust_sched_launcher *create(rust_scheduler *sched, int id,
                                        bool killed);
    rust_sched_driver *get_driver() {
        assert(launcher != NULL);
        return launcher->get_driver();
    }
};

#endif // RUST_SCHED_LAUNCHER_H
