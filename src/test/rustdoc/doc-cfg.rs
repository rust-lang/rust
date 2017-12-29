// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(doc_cfg)]

// @has doc_cfg/struct.Portable.html
// @!has - '//*[@id="main"]/*[@class="stability"]/*[@class="stab portability"]' ''
// @has - '//*[@id="method.unix_and_arm_only_function"]' 'fn unix_and_arm_only_function()'
// @has - '//*[@class="stab portability"]' 'This is supported on Unix and ARM only.'
pub struct Portable;

// @has doc_cfg/unix_only/index.html \
//  '//*[@id="main"]/*[@class="stability"]/*[@class="stab portability"]' \
//  'This is supported on Unix only.'
// @matches - '//*[@class=" module-item"]//*[@class="stab portability"]' '\AUnix\Z'
// @matches - '//*[@class=" module-item"]//*[@class="stab portability"]' '\AUnix and ARM\Z'
// @count - '//*[@class="stab portability"]' 3
#[doc(cfg(unix))]
pub mod unix_only {
    // @has doc_cfg/unix_only/fn.unix_only_function.html \
    //  '//*[@id="main"]/*[@class="stability"]/*[@class="stab portability"]' \
    //  'This is supported on Unix only.'
    // @count - '//*[@class="stab portability"]' 1
    pub fn unix_only_function() {
        content::should::be::irrelevant();
    }

    // @has doc_cfg/unix_only/trait.ArmOnly.html \
    //  '//*[@id="main"]/*[@class="stability"]/*[@class="stab portability"]' \
    //  'This is supported on Unix and ARM only.'
    // @count - '//*[@class="stab portability"]' 2
    #[doc(cfg(target_arch = "arm"))]
    pub trait ArmOnly {
        fn unix_and_arm_only_function();
    }

    impl ArmOnly for super::Portable {
        fn unix_and_arm_only_function() {}
    }
}
