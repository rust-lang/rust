// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <assert.h>
#include <setjmp.h>

static jmp_buf ENV;

extern void test_middle();

void test_start(void(*f)()) {
  if (setjmp(ENV) != 0)
    return;
  f();
  assert(0);
}

void test_end() {
  longjmp(ENV, 1);
  assert(0);
}
