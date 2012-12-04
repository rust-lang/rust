// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(ctypes)];
#[allow(heap_memory)];
#[allow(implicit_copies)];
#[allow(managed_heap_memory)];
#[allow(non_camel_case_types)];
#[allow(non_implicitly_copyable_typarams)];
#[allow(owned_heap_memory)];
#[allow(path_statement)];
#[allow(structural_records)];
#[allow(unrecognized_lint)];
#[allow(unused_imports)];
#[allow(vecs_implicitly_copyable)];
#[allow(while_true)];

extern mod std;

fn print<T>(result: T) {
    io::println(fmt!("%?", result));
}
