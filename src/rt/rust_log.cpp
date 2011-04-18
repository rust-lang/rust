/*
 * Logging infrastructure that aims to support multi-threading, indentation
 * and ansi colors.
 */

#include "rust_internal.h"
#include "util/array_list.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

// FIXME somehow unify this with the parsing happening in rust_crate.cpp
static uint32_t
read_type_bit_mask() {
    uint32_t bits = rust_log::ULOG | rust_log::ERR;
    char *env_str = getenv("RUST_LOG");
    if (env_str) {
        char *str = new char[strlen(env_str) + 2];
        str[0] = ',';
        strcpy(str + 1, env_str);

        bits = rust_log::ULOG;
        bits |= strstr(str, ",err") ? rust_log::ERR : 0;
        bits |= strstr(str, ",mem") ? rust_log::MEM : 0;
        bits |= strstr(str, ",comm") ? rust_log::COMM : 0;
        bits |= strstr(str, ",task") ? rust_log::TASK : 0;
        bits |= strstr(str, ",up") ? rust_log::UPCALL : 0;
        bits |= strstr(str, ",dom") ? rust_log::DOM : 0;
        bits |= strstr(str, ",trace") ? rust_log::TRACE : 0;
        bits |= strstr(str, ",dwarf") ? rust_log::DWARF : 0;
        bits |= strstr(str, ",cache") ? rust_log::CACHE : 0;
        bits |= strstr(str, ",timer") ? rust_log::TIMER : 0;
        bits |= strstr(str, ",gc") ? rust_log::GC : 0;
        bits |= strstr(str, ",stdlib") ? rust_log::STDLIB : 0;
        bits |= strstr(str, ",special") ? rust_log::SPECIAL : 0;
        bits |= strstr(str, ",kern") ? rust_log::KERN : 0;
        bits |= strstr(str, ",bt") ? rust_log::BT : 0;
        bits |= strstr(str, ",all") ? rust_log::ALL : 0;
        bits = strstr(str, ",none") ? 0 : bits;

        delete[] str;
    }
    return bits;
}

rust_log::ansi_color
get_type_color(rust_log::log_type type) {
    rust_log::ansi_color color = rust_log::WHITE;
    if (type & rust_log::ERR)
        color = rust_log::RED;
    if (type & rust_log::MEM)
        color = rust_log::YELLOW;
    if (type & rust_log::UPCALL)
        color = rust_log::GREEN;
    if (type & rust_log::COMM)
        color = rust_log::MAGENTA;
    if (type & rust_log::DOM)
        color = rust_log::LIGHTTEAL;
    if (type & rust_log::TASK)
        color = rust_log::LIGHTTEAL;
    return color;
}

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
    _type_bit_mask(read_type_bit_mask()),
    _use_colors(getenv("RUST_COLOR_LOG")),
    _indent(0) {
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
    for (uint32_t i = 0; i < _indent; i++) {
        append_string(buffer, "    ");
    }
    append_string(buffer, "%s", message);
    if (_last_thread_id != thread_id) {
        _last_thread_id = thread_id;
        _srv->log("---");
    }
    _srv->log(buffer);
    _log_lock.unlock();
}

void
rust_log::trace_ln(rust_task *task, char *message) {
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

/**
 * Traces a log message if the specified logging type is not filtered.
 */
void
rust_log::trace_ln(rust_task *task, uint32_t type_bits, char *message) {
    trace_ln(task, get_type_color((rust_log::log_type) type_bits),
             type_bits, message);
}

/**
 * Traces a log message using the specified ANSI color code.
 */
void
rust_log::trace_ln(rust_task *task, ansi_color color,
                   uint32_t type_bits, char *message) {
    if (is_tracing(type_bits)) {
        if (_use_colors) {
            char buffer[BUF_BYTES] = "";
            append_string(buffer, color, "%s", message);
            trace_ln(task, buffer);
        } else {
            trace_ln(task, message);
        }
    }
}

void
rust_log::indent() {
    _indent++;
}

void
rust_log::outdent() {
    _indent--;
}

void
rust_log::reset_indent(uint32_t indent) {
    _indent = indent;
}

struct mod_entry {
    const char* name;
    int* state;
};

struct cratemap {
    mod_entry* entries;
    cratemap* children[1];
};

struct log_directive {
    char* name;
    size_t level;
};

const size_t max_log_directives = 255;

// This is a rather ugly parser for strings in the form
// "crate1,crate2.mod3,crate3.x=2". Log levels range 0=err, 1=warn, 2=info,
// 3=debug. Default is 1. Words without an '=X' part set the log level for
// that module (and submodules) to 3.
size_t parse_logging_spec(char* spec, log_directive* dirs) {
    size_t dir = 0;
    while (dir < max_log_directives && *spec) {
        char* start = spec;
        char cur;
        while (true) {
            cur = *spec;
            if (cur == ',' || cur == '=' || cur == '\0') {
                if (start == spec) {spec++; break;}
                *spec = '\0';
                spec++;
                size_t level = 3;
                if (cur == '=') {
                    level = *spec - '0';
                    if (level > 3) level = 1;
                    if (*spec) ++spec;
                }
                dirs[dir].name = start;
                dirs[dir++].level = level;
                break;
            }
            spec++;
        }
    }
    return dir;
}

void update_crate_map(cratemap* map, log_directive* dirs, size_t n_dirs) {
    // First update log levels for this crate
    for (mod_entry* cur = map->entries; cur->name; cur++) {
        size_t level = 1, longest_match = 0;
        for (size_t d = 0; d < n_dirs; d++) {
            if (strstr(cur->name, dirs[d].name) == cur->name &&
                strlen(dirs[d].name) > longest_match) {
                longest_match = strlen(dirs[d].name);
                level = dirs[d].level;
            }
        }
        *cur->state = level;
    }

    // Then recurse on linked crates
    for (size_t i = 0; map->children[i]; i++) {
        update_crate_map(map->children[i], dirs, n_dirs);
    }
}

void update_log_settings(void* crate_map, char* settings) {
    char* buffer = NULL;
    log_directive dirs[256];
    size_t dir = 0;
    if (settings) {
        buffer = (char*)malloc(strlen(settings));
        strcpy(buffer, settings);
        dir = parse_logging_spec(buffer, &dirs[0]);
    }

    update_crate_map((cratemap*)crate_map, &dirs[0], dir);

    free(buffer);
}
