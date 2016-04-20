#!/usr/bin/env python
#
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os
import shutil
import sys
import tarfile

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bootstrap"))
sys.path.append(path)

import bootstrap

def main(argv):
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    triple = argv[1]
    data = bootstrap.stage0_data(src_root)

    channel, date = data['rustc'].split('-', 1)

    dl_dir = 'dl'
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)

    filename_base = 'rustc-' + channel + '-' + triple
    filename = filename_base + '.tar.gz'
    url = 'https://static.rust-lang.org/dist/' + date + '/' + filename
    dst = dl_dir + '/' + filename
    if not os.path.exists(dst):
        bootstrap.get(url, dst)

    stage0_dst = triple + '/stage0'
    if os.path.exists(stage0_dst):
        for root, _, files in os.walk(stage0_dst):
            for f in files:
                os.unlink(os.path.join(root, f))
    else:
        os.makedirs(stage0_dst)
    bootstrap.unpack(dst, stage0_dst, match='rustc', verbose=True)

if __name__ == '__main__':
    main(sys.argv)
