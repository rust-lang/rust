// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
 * Logging infrastructure that aims to support multi-threading
 */


#include "rust_log.h"
#include "rust_crate_map.h"
#include "util/array_list.h"
#include "rust_util.h"
#include "rust_task.h"

/**
 * Synchronizes access to the underlying logging mechanism.
 */
static lock_and_signal _log_lock;
/**
 * Indicates whether we are outputing to the console.
 * Protected by _log_lock;
 */
static bool _log_to_console = true;

/*
 * Request that console logging be turned on.
 */
void
log_console_on() {
    scoped_lock with(_log_lock);
    _log_to_console = true;
}

/*
 * Request that console logging be turned off. Can be
 * overridden by the environment.
 */
void
log_console_off(rust_env *env) {
    scoped_lock with(_log_lock);
    if (env->logspec == NULL) {
        _log_to_console = false;
    }
}

rust_log::rust_log(rust_sched_loop *sched_loop) :
    _sched_loop(sched_loop) {
}

rust_log::~rust_log() {

}

const uint16_t
hash(uintptr_t ptr) {
#   if(ULONG_MAX == 0xFFFFFFFF)
    // Robert Jenkins' 32 bit integer hash function
    ptr = (ptr + 0x7ed55d16) + (ptr << 12);
    ptr = (ptr ^ 0xc761c23c) ^ (ptr >> 19);
    ptr = (ptr + 0x165667b1) + (ptr << 5);
    ptr = (ptr + 0xd3a2646c) ^ (ptr << 9);
    ptr = (ptr + 0xfd7046c5) + (ptr << 3);
    ptr = (ptr ^ 0xb55a4f09) ^ (ptr >> 16);
#   elif(ULONG_MAX == 0xFFFFFFFFFFFFFFFF)
    // "hash64shift()" from http://www.concentric.net/~Ttwang/tech/inthash.htm
    ptr = (~ptr) + (ptr << 21); // ptr = (ptr << 21) - ptr - 1;
    ptr = ptr ^ (ptr >> 24);
    ptr = (ptr + (ptr << 3)) + (ptr << 8); // ptr * 265
    ptr = ptr ^ (ptr >> 14);
    ptr = (ptr + (ptr << 2)) + (ptr << 4); // ptr * 21
    ptr = ptr ^ (ptr >> 28);
    ptr = ptr + (ptr << 31);
#   else
#   error "hash() not defined for this pointer size"
#   endif
    return (uint16_t) ptr;
}

char *
copy_string(char *dst, const char *src, size_t length) {
    return strncpy(dst, src, length) + length;
}

char *
append_string(char *buffer, const char *format, ...) {
    if (buffer != NULL && format) {
        va_list args;
        va_start(args, format);
        size_t off = strlen(buffer);
        vsnprintf(buffer + off, BUF_BYTES - off, format, args);
        va_end(args);
    }
    return buffer;
}

void
rust_log::log(rust_task* task, uint32_t level, char const *fmt, ...) {
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, fmt);
    int formattedbytes = vsnprintf(buf, sizeof(buf), fmt, args);
    if( formattedbytes and (unsigned)formattedbytes > BUF_BYTES ){
        const char truncatedstr[] = "[...]";
        memcpy( &buf[BUF_BYTES-sizeof(truncatedstr)],
                truncatedstr,
                sizeof(truncatedstr));
    }
    trace_ln(task, level, buf);
    va_end(args);
}

void
rust_log::trace_ln(char *prefix, char *message) {
    char buffer[BUF_BYTES] = "";
    _log_lock.lock();
    append_string(buffer, "%s", prefix);
    append_string(buffer, "%s", message);
    if (_log_to_console) {
        fprintf(stderr, "rust: %s\n", buffer);
        fflush(stderr);
    }
    _log_lock.unlock();
}

void
rust_log::trace_ln(rust_task *task, uint32_t level, char *message) {

    if (task) {
        // There is not enough room to be logging on the rust stack
        assert(!task->on_rust_stack() && "logging on rust stack");
    }

    // FIXME (#2672): The scheduler and task names used to have meaning,
    // but they are always equal to 'main' currently
#if 0

#if defined(__WIN32__)
    uint32_t thread_id = 0;
#else
    uint32_t thread_id = hash((uintptr_t) pthread_self());
#endif

    char prefix[BUF_BYTES] = "";
    if (_sched_loop && _sched_loop-.name) {
        append_string(prefix, "%04" PRIxPTR ":%.10s:",
                      thread_id, _sched_loop->name);
    } else {
        append_string(prefix, "%04" PRIxPTR ":0x%08" PRIxPTR ":",
                      thread_id, (uintptr_t) _sched_loop);
    }
    if (task) {
        if (task->name) {
            append_string(prefix, "%.10s:", task->name);
        } else {
            append_string(prefix, "0x%08" PRIxPTR ":", (uintptr_t) task);
        }
    }
#else
    char prefix[BUF_BYTES] = "";
#endif

    trace_ln(prefix, message);
}

// Reading log directives and setting log level vars

struct log_directive {
    char* name;
    size_t level;
};

const size_t max_log_directives = 255;
const size_t max_log_level = 255;
const size_t default_log_level = log_err;

// This is a rather ugly parser for strings in the form
// "crate1,crate2.mod3,crate3.x=1". Log levels are 0-255,
// with the most likely ones being 0-3 (defined in core::).
size_t parse_logging_spec(char* spec, log_directive* dirs) {
    size_t dir = 0;
    while (dir < max_log_directives && *spec) {
        char* start = spec;
        char cur;
        while (true) {
            cur = *spec;
            if (cur == ',' || cur == '=' || cur == '\0') {
                if (start == spec) {spec++; break;}
                if (*spec != '\0') {
                    *spec = '\0';
                    spec++;
                }
                size_t level = max_log_level;
                if (cur == '=' && *spec != '\0') {
                    level = *spec - '0';
                    if (level > max_log_level) level = max_log_level;
                    if (*spec) ++spec;
                }
                dirs[dir].name = start;
                dirs[dir++].level = level;
                break;
            } else {
                spec++;
            }
        }
    }
    return dir;
}

struct update_entry_args {
    log_directive* dirs;
    size_t n_dirs;
    size_t *n_matches;
};

static void update_entry(const mod_entry* entry, void *cookie) {
    update_entry_args *args = (update_entry_args *)cookie;
    size_t level = default_log_level, longest_match = 0;
    for (size_t d = 0; d < args->n_dirs; d++) {
        if (strstr(entry->name, args->dirs[d].name) == entry->name &&
            strlen(args->dirs[d].name) > longest_match) {
            longest_match = strlen(args->dirs[d].name);
            level = args->dirs[d].level;
        }
    }
    *entry->state = level;
    (*args->n_matches)++;
}

void update_module_map(const mod_entry* map, log_directive* dirs,
                       size_t n_dirs, size_t *n_matches) {
    update_entry_args args = { dirs, n_dirs, n_matches };
    iter_module_map(map, update_entry, &args);
}

void update_crate_map(const cratemap* map, log_directive* dirs,
                      size_t n_dirs, size_t *n_matches) {
    update_entry_args args = { dirs, n_dirs, n_matches };
    iter_crate_map(map, update_entry, &args);
}

void print_mod_name(const mod_entry* mod, void *cooke) {
    printf(" %s\n", mod->name);
}

void print_crate_log_map(const cratemap* map) {
    iter_crate_map(map, print_mod_name, NULL);
}

// These are pseudo-modules used to control logging in the runtime.

uint32_t log_rt_mem;
uint32_t log_rt_box;
uint32_t log_rt_comm;
uint32_t log_rt_task;
uint32_t log_rt_dom;
uint32_t log_rt_trace;
uint32_t log_rt_cache;
uint32_t log_rt_upcall;
uint32_t log_rt_timer;
uint32_t log_rt_gc;
uint32_t log_rt_stdlib;
uint32_t log_rt_kern;
uint32_t log_rt_backtrace;
uint32_t log_rt_callback;

static const mod_entry _rt_module_map[] =
    {{"::rt::mem", &log_rt_mem},
     {"::rt::box", &log_rt_box},
     {"::rt::comm", &log_rt_comm},
     {"::rt::task", &log_rt_task},
     {"::rt::dom", &log_rt_dom},
     {"::rt::trace", &log_rt_trace},
     {"::rt::cache", &log_rt_cache},
     {"::rt::upcall", &log_rt_upcall},
     {"::rt::timer", &log_rt_timer},
     {"::rt::gc", &log_rt_gc},
     {"::rt::stdlib", &log_rt_stdlib},
     {"::rt::kern", &log_rt_kern},
     {"::rt::backtrace", &log_rt_backtrace},
     {"::rt::callback", &log_rt_callback},
     {NULL, NULL}};

void update_log_settings(void* crate_map, char* settings) {
    char* buffer = NULL;
    log_directive dirs[256];
    size_t n_dirs = 0;

    if (settings) {

        if (strcmp(settings, "::help") == 0 ||
            strcmp(settings, "?") == 0) {
            printf("\nCrate log map:\n\n");
            print_crate_log_map((const cratemap*)crate_map);
            printf("\n");
            exit(1);
        }

        size_t buflen = strlen(settings) + 1;
        buffer = (char*)malloc(buflen);
        strncpy(buffer, settings, buflen);
        n_dirs = parse_logging_spec(buffer, &dirs[0]);
    }

    size_t n_matches = 0;
    update_module_map(_rt_module_map, &dirs[0], n_dirs, &n_matches);
    update_crate_map((const cratemap*)crate_map, &dirs[0],
                     n_dirs, &n_matches);

    if (n_matches < n_dirs) {
        // NB: Android compiler is complaining about format specifiers here
        // and I don't understand why
        /*printf("warning: got %" PRIdPTR " RUST_LOG specs, "
               "enabled %" PRIdPTR " flags.",
               (uintptr_t)n_dirs, (uintptr_t)n_matches);*/
    }

    free(buffer);
}


//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
