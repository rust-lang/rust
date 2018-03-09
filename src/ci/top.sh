#!/usr/bin/env bash
# Copyright 2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

while `sleep 30`
do
# get tops summary and print it into a logfile every 30 secs
# run top in batch mode (b), skip idle processes (i) and quite after first output (n 1)
top -ibn 1 | head -n4 | tr "\n" " " | tee -a /tmp/top.log
echo "" | tee -a /tmp/top.log
done
