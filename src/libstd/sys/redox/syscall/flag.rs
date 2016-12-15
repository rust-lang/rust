pub const CLONE_VM: usize = 0x100;
pub const CLONE_FS: usize = 0x200;
pub const CLONE_FILES: usize = 0x400;
pub const CLONE_VFORK: usize = 0x4000;

pub const CLOCK_REALTIME: usize = 1;
pub const CLOCK_MONOTONIC: usize = 4;

pub const EVENT_NONE: usize = 0;
pub const EVENT_READ: usize = 1;
pub const EVENT_WRITE: usize = 2;

pub const F_GETFL: usize = 1;
pub const F_SETFL: usize = 2;

pub const FUTEX_WAIT: usize = 0;
pub const FUTEX_WAKE: usize = 1;
pub const FUTEX_REQUEUE: usize = 2;

pub const MAP_WRITE: usize = 1;
pub const MAP_WRITE_COMBINE: usize = 2;

pub const MODE_TYPE: u16 = 0xF000;
pub const MODE_DIR: u16 = 0x4000;
pub const MODE_FILE: u16 = 0x8000;

pub const MODE_PERM: u16 = 0x0FFF;
pub const MODE_SETUID: u16 = 0o4000;
pub const MODE_SETGID: u16 = 0o2000;

pub const O_RDONLY: usize =     0x0001_0000;
pub const O_WRONLY: usize =     0x0002_0000;
pub const O_RDWR: usize =       0x0003_0000;
pub const O_NONBLOCK: usize =   0x0004_0000;
pub const O_APPEND: usize =     0x0008_0000;
pub const O_SHLOCK: usize =     0x0010_0000;
pub const O_EXLOCK: usize =     0x0020_0000;
pub const O_ASYNC: usize =      0x0040_0000;
pub const O_FSYNC: usize =      0x0080_0000;
pub const O_CLOEXEC: usize =    0x0100_0000;
pub const O_CREAT: usize =      0x0200_0000;
pub const O_TRUNC: usize =      0x0400_0000;
pub const O_EXCL: usize =       0x0800_0000;
pub const O_DIRECTORY: usize =  0x1000_0000;
pub const O_STAT: usize =       0x2000_0000;
pub const O_ACCMODE: usize =    O_RDONLY | O_WRONLY | O_RDWR;

pub const SEEK_SET: usize = 0;
pub const SEEK_CUR: usize = 1;
pub const SEEK_END: usize = 2;

pub const SIGHUP: usize =   1;
pub const SIGINT: usize =   2;
pub const SIGQUIT: usize =  3;
pub const SIGILL: usize =   4;
pub const SIGTRAP: usize =  5;
pub const SIGABRT: usize =  6;
pub const SIGBUS: usize =   7;
pub const SIGFPE: usize =   8;
pub const SIGKILL: usize =  9;
pub const SIGUSR1: usize =  10;
pub const SIGSEGV: usize =  11;
pub const SIGUSR2: usize =  12;
pub const SIGPIPE: usize =  13;
pub const SIGALRM: usize =  14;
pub const SIGTERM: usize =  15;
pub const SIGSTKFLT: usize= 16;
pub const SIGCHLD: usize =  17;
pub const SIGCONT: usize =  18;
pub const SIGSTOP: usize =  19;
pub const SIGTSTP: usize =  20;
pub const SIGTTIN: usize =  21;
pub const SIGTTOU: usize =  22;
pub const SIGURG: usize =   23;
pub const SIGXCPU: usize =  24;
pub const SIGXFSZ: usize =  25;
pub const SIGVTALRM: usize= 26;
pub const SIGPROF: usize =  27;
pub const SIGWINCH: usize = 28;
pub const SIGIO: usize =    29;
pub const SIGPWR: usize =   30;
pub const SIGSYS: usize =   31;

pub const WNOHANG: usize = 1;
