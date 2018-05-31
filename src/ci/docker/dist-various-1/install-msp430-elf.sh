# Copyright 2018 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

mkdir /usr/local/msp430-none-elf

# Newer versions of toolchain can be found here
# http://software-dl.ti.com/msp430/msp430_public_sw/mcu/msp430/MSPGCC/latest/index_FDS.html
# Original link for version 5_01_02_00 (6.4.0.32) is
# http://software-dl.ti.com/msp430/msp430_public_sw/mcu/msp430/MSPGCC/5_01_02_00/exports/msp430-gcc-6.4.0.32_linux64.tar.bz2
# TI website doesn't allow curl, so we have to use mirror
URL="https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror"
FILE="msp430-gcc-6.4.0.32_linux64.tar.bz2"
curl -L "$URL/$FILE" | tar xjf - -C /usr/local/msp430-none-elf --strip-components=1

for file in /usr/local/msp430-none-elf/bin/msp430-elf-*; do
  ln -s $file /usr/local/bin/`basename $file`
done
