/*
 * Logging infrastructure that aims to support multi-threading,
 * and ansi colors.
 */

#include "rust_internal.h"
#include "util/array_list.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

static const char * _foreground_colors[] = { "[37m",
                                             "[31m", "[1;31m",
                                             "[32m", "[1;32m",
                                             "[33m", "[1;33m",
                                             "[31m", "[1;31m",
                                             "[35m", "[1;35m",
                                             "[36m", "[1;36m" };

/**
 * Synchronizes access to the underlying logging mechanism.
 */
static lock_and_signal _log_lock;
static uint32_t _last_thread_id;

rust_log::rust_log(rust_srv *srv, rust_dom *dom) :
    _srv(srv),
    _dom(dom),
    _use_colors(getenv("RUST_COLOR_LOG")) {
}

rust_log::~rust_log() {

}

const uint16_t
hash(uintptr_t ptr) {
    // Robert Jenkins' 32 bit integer hash function
    ptr = (ptr + 0x7ed55d16) + (ptr << 12);
    ptr = (ptr ^ 0xc761c23c) ^ (ptr >> 19);
    ptr = (ptr + 0x165667b1) + (ptr << 5);
    ptr = (ptr + 0xd3a2646c) ^ (ptr << 9);
    ptr = (ptr + 0xfd7046c5) + (ptr << 3);
    ptr = (ptr ^ 0xb55a4f09) ^ (ptr >> 16);
    return (uint16_t) ptr;
}

const char *
get_color(uintptr_t ptr) {
    return _foreground_colors[hash(ptr) % rust_log::LIGHTTEAL];
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

char *
append_string(char *buffer, rust_log::ansi_color color,
              const char *format, ...) {
    if (buffer != NULL && format) {
        append_string(buffer, "\x1b%s", _foreground_colors[color]);
        va_list args;
        va_start(args, format);
        size_t off = strlen(buffer);
        vsnprintf(buffer + off, BUF_BYTES - off, format, args);
        va_end(args);
        append_string(buffer, "\x1b[0m");
    }
    return buffer;
}

void
rust_log::trace_ln(uint32_t thread_id, char *prefix, char *message) {
    char buffer[BUF_BYTES] = "";
    _log_lock.lock();
    append_string(buffer, "%-34s", prefix);
    append_string(buffer, "%s", message);
    if (_last_thread_id != thread_id) {
        _last_thread_id = thread_id;
        _srv->log("---");
    }
    _srv->log(buffer);
    _log_lock.unlock();
}

void
rust_log::trace_ln(rust_task *task, uint32_t level, char *message) {
#if defined(__WIN32__)
    uint32_t thread_id = 0;
#else
    uint32_t thread_id = hash((uint32_t) pthread_self());
#endif
    char prefix[BUF_BYTES] = "";
    if (_dom && _dom->name) {
        append_string(prefix, "%04" PRIxPTR ":%.10s:",
                      thread_id, _dom->name);
    } else {
        append_string(prefix, "%04" PRIxPTR ":0x%08" PRIxPTR ":",
                      thread_id, (uintptr_t) _dom);
    }
    if (task) {
        if (task->name) {
            append_string(prefix, "%.10s:", task->name);
        } else {
            append_string(prefix, "0x%08" PRIxPTR ":", (uintptr_t) task);
        }
    }
    trace_ln(thread_id, prefix, message);
}

// Reading log directives and setting log level vars

struct mod_entry {
    const char* name;
    size_t* state;
};

struct cratemap {
    const mod_entry* entries;
    const cratemap* children[1];
};

struct log_directive {
    char* name;
    size_t level;
};

const size_t max_log_directives = 255;
const size_t max_log_level = 1;
const size_t default_log_level = 0;

// This is a rather ugly parser for strings in the form
// "crate1,crate2.mod3,crate3.x=1". Log levels are 0-1 for now,
// eventually we'll have 0-3.
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

void update_module_map(const mod_entry* map, log_directive* dirs,
                       size_t n_dirs) {
    for (const mod_entry* cur = map; cur->name; cur++) {
        size_t level = default_log_level, longest_match = 0;
        for (size_t d = 0; d < n_dirs; d++) {
            if (strstr(cur->name, dirs[d].name) == cur->name &&
                strlen(dirs[d].name) > longest_match) {
                longest_match = strlen(dirs[d].name);
                level = dirs[d].level;
            }
        }
        *cur->state = level;
    }
}

void update_crate_map(const cratemap* map, log_directive* dirs,
                      size_t n_dirs) {
    // First update log levels for this crate
    update_module_map(map->entries, dirs, n_dirs);
    // Then recurse on linked crates
    // FIXME this does double work in diamond-shaped deps. could keep
    //   a set of visited addresses, if it turns out to be actually slow
    for (size_t i = 0; map->children[i]; i++) {
        update_crate_map(map->children[i], dirs, n_dirs);
    }
}

// These are pseudo-modules used to control logging in the runtime.

size_t log_rt_mem;
size_t log_rt_comm;
size_t log_rt_task;
size_t log_rt_dom;
size_t log_rt_trace;
size_t log_rt_cache;
size_t log_rt_upcall;
size_t log_rt_timer;
size_t log_rt_gc;
size_t log_rt_stdlib;
size_t log_rt_kern;
size_t log_rt_backtrace;

static const mod_entry _rt_module_map[] =
    {{"rt.mem", &log_rt_mem},
     {"rt.comm", &log_rt_comm},
     {"rt.task", &log_rt_task},
     {"rt.dom", &log_rt_dom},
     {"rt.trace", &log_rt_trace},
     {"rt.cache", &log_rt_cache},
     {"rt.upcall", &log_rt_upcall},
     {"rt.timer", &log_rt_timer},
     {"rt.gc", &log_rt_gc},
     {"rt.stdlib", &log_rt_stdlib},
     {"rt.kern", &log_rt_kern},
     {"rt.backtrace", &log_rt_backtrace},
     {NULL, NULL}};

void update_log_settings(void* crate_map, char* settings) {
    char* buffer = NULL;
    log_directive dirs[256];
    size_t n_dirs = 0;
    if (settings) {
        size_t buflen = strlen(settings) + 1;
        buffer = (char*)malloc(buflen);
        strncpy(buffer, settings, buflen);
        n_dirs = parse_logging_spec(buffer, &dirs[0]);
    }

    update_module_map(_rt_module_map, &dirs[0], n_dirs);
    update_crate_map((const cratemap*)crate_map, &dirs[0], n_dirs);

    free(buffer);
}
