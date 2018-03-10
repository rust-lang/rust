#!/usr/bin/env bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

wget http://packages.haiku-os.org/haikuports/master/hpkg/llvm-4.0.1-2-x86_64.hpkg
wget http://packages.haiku-os.org/haikuports/master/hpkg/llvm_libs-4.0.1-2-x86_64.hpkg

package extract -C /system llvm-4.0.1-2-x86_64.hpkg
package extract -C /system llvm_libs-4.0.1-2-x86_64.hpkg

rm -f *.hpkg
