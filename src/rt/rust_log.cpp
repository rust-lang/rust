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


#include "rust_crate_map.h"
#include "util/array_list.h"
#include "rust_util.h"

// Reading log directives and setting log level vars

struct log_directive {
    char* name;
    size_t level;
};


const uint32_t log_err = 1;
const uint32_t log_warn = 2;
const uint32_t log_info = 3;
const uint32_t log_debug = 4;

const size_t max_log_directives = 255;
const size_t max_log_level = 255;
const size_t default_log_level = log_err;

// This is a rather ugly parser for strings in the form
// "crate1,crate2.mod3,crate3.x=1". Log levels are 0-255,
// with the most likely ones being 0-3 (defined in std::).
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
    if (longest_match > 0) {
        (*args->n_matches)++;
    }
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
    update_crate_map((const cratemap*)crate_map, &dirs[0],
                     n_dirs, &n_matches);

    if (n_matches < n_dirs) {
        fprintf(stderr, "warning: got %lu RUST_LOG specs but only matched %lu of them.\n"
                "You may have mistyped a RUST_LOG spec.\n"
                "Use RUST_LOG=::help to see the list of crates and modules.\n",
                (unsigned long)n_dirs, (unsigned long)n_matches);
    }

    free(buffer);
}

extern "C" CDECL void
rust_update_log_settings(void* crate_map, char* settings) {
    update_log_settings(crate_map, settings);
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
