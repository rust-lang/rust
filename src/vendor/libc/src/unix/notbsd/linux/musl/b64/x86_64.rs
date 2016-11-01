pub type c_char = i8;

s! {
    pub struct mcontext_t {
        __private: [u64; 32],
    }

    pub struct ucontext_t {
        pub uc_flags: ::c_ulong,
        pub uc_link: *mut ucontext_t,
        pub uc_stack: ::stack_t,
        pub uc_mcontext: mcontext_t,
        pub uc_sigmask: ::sigset_t,
        __private: [u8; 512],
    }
}

pub const SYS_gettid: ::c_long = 186;

pub const SYS_perf_event_open: ::c_long = 298;
