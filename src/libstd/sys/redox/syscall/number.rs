// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub const SYS_CLASS: usize =    0xF000_0000;
pub const SYS_CLASS_PATH: usize=0x1000_0000;
pub const SYS_CLASS_FILE: usize=0x2000_0000;

pub const SYS_ARG: usize =      0x0F00_0000;
pub const SYS_ARG_SLICE: usize =0x0100_0000;
pub const SYS_ARG_MSLICE: usize=0x0200_0000;
pub const SYS_ARG_PATH: usize = 0x0300_0000;

pub const SYS_RET: usize =      0x00F0_0000;
pub const SYS_RET_FILE: usize = 0x0010_0000;

pub const SYS_LINK: usize =     SYS_CLASS_PATH | SYS_ARG_PATH | 9;
pub const SYS_OPEN: usize =     SYS_CLASS_PATH | SYS_RET_FILE | 5;
pub const SYS_CHMOD: usize =    SYS_CLASS_PATH | 15;
pub const SYS_RMDIR: usize =    SYS_CLASS_PATH | 84;
pub const SYS_UNLINK: usize =   SYS_CLASS_PATH | 10;

pub const SYS_CLOSE: usize =    SYS_CLASS_FILE | 6;
pub const SYS_DUP: usize =      SYS_CLASS_FILE | SYS_RET_FILE | 41;
pub const SYS_READ: usize =     SYS_CLASS_FILE | SYS_ARG_MSLICE | 3;
pub const SYS_WRITE: usize =    SYS_CLASS_FILE | SYS_ARG_SLICE | 4;
pub const SYS_LSEEK: usize =    SYS_CLASS_FILE | 19;
pub const SYS_FCNTL: usize =    SYS_CLASS_FILE | 55;
pub const SYS_FEVENT: usize =   SYS_CLASS_FILE | 927;
pub const SYS_FMAP: usize =     SYS_CLASS_FILE | 90;
pub const SYS_FUNMAP: usize =   SYS_CLASS_FILE | 91;
pub const SYS_FPATH: usize =    SYS_CLASS_FILE | SYS_ARG_MSLICE | 928;
pub const SYS_FSTAT: usize =    SYS_CLASS_FILE | SYS_ARG_MSLICE | 28;
pub const SYS_FSTATVFS: usize = SYS_CLASS_FILE | SYS_ARG_MSLICE | 100;
pub const SYS_FSYNC: usize =    SYS_CLASS_FILE | 118;
pub const SYS_FTRUNCATE: usize =SYS_CLASS_FILE | 93;

pub const SYS_BRK: usize =      45;
pub const SYS_CHDIR: usize =    12;
pub const SYS_CLOCK_GETTIME: usize = 265;
pub const SYS_CLONE: usize =    120;
pub const SYS_EXECVE: usize =   11;
pub const SYS_EXIT: usize =     1;
pub const SYS_FUTEX: usize =    240;
pub const SYS_GETCWD: usize =   183;
pub const SYS_GETEGID: usize =  202;
pub const SYS_GETENS: usize =   951;
pub const SYS_GETEUID: usize =  201;
pub const SYS_GETGID: usize =   200;
pub const SYS_GETNS: usize =    950;
pub const SYS_GETPID: usize =   20;
pub const SYS_GETUID: usize =   199;
pub const SYS_IOPL: usize =     110;
pub const SYS_KILL: usize =     37;
pub const SYS_MKNS: usize =     984;
pub const SYS_NANOSLEEP: usize =162;
pub const SYS_PHYSALLOC: usize =945;
pub const SYS_PHYSFREE: usize = 946;
pub const SYS_PHYSMAP: usize =  947;
pub const SYS_PHYSUNMAP: usize =948;
pub const SYS_VIRTTOPHYS: usize=949;
pub const SYS_PIPE2: usize =    331;
pub const SYS_SETREGID: usize = 204;
pub const SYS_SETRENS: usize =  952;
pub const SYS_SETREUID: usize = 203;
pub const SYS_WAITPID: usize =  7;
pub const SYS_YIELD: usize =    158;
