// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions -Z emit-end-regions
// ignore-tidy-linelength

// We must mark a variable whose initialization fails due to an
// abort statement as StorageDead.

fn main() {
    loop {
        let beacon = {
            match true {
                false => 4,
                true => break,
            }
        };
        drop(&beacon);
    }
}

// END RUST SOURCE
// START rustc.main.mir_map.0.mir
//    ...
//    scope 1 {
//        let _2: i32;
//    }
//    ...
//    bb3: {
//        StorageLive(_2);
//        StorageLive(_3);
//        _3 = const true;
//        EndRegion('3s);
//        _4 = discriminant(_3);
//        switchInt(_3) -> [false: bb11, otherwise: bb10];
//    }
//    ...
//    bb22: {
//        EndRegion('20_0rs);
//        StorageDead(_2);
//        goto -> bb23;
//    }
//    ...
//    bb28: {
//        EndRegion('18s);
//        StorageDead(_7);
//        EndRegion('19s);
//        EndRegion('19ds);
//        _1 = ();
//        EndRegion('20_0rs);
//        StorageDead(_2);
//        EndRegion('20s);
//        EndRegion('20ds);
//        goto -> bb1;
//    }
//    bb29: {
//        return;
//    }
// END rustc.main.mir_map.0.mir
