// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rust_exchange_alloc.h"
#include "sync/sync.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

void *
rust_exchange_alloc::malloc(size_t size) {
  void *value = ::malloc(size);
  assert(value);
  return value;
}

void *
rust_exchange_alloc::realloc(void *ptr, size_t size) {
  void *new_ptr = ::realloc(ptr, size);
  assert(new_ptr);
  return new_ptr;
}

void
rust_exchange_alloc::free(void *ptr) {
  ::free(ptr);
}
