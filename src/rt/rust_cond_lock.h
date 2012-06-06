// -*- c++ -*-
// A lock and condition variable pair that is useable from Rust.

#pragma once

#include "sync/lock_and_signal.h"
#include "rust_globals.h"
#include "rust_task.h"

struct rust_cond_lock : public rust_cond {
    rust_cond_lock();

    lock_and_signal lock;
    rust_task *waiting;
};
