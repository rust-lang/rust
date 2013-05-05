// xfail-license

#define RUSTRT_RBX   0
#define RUSTRT_RSP   1
#define RUSTRT_RBP   2
// RCX on Windows, RDI elsewhere
#define RUSTRT_ARG0  3
#define RUSTRT_R12   4
#define RUSTRT_R13   5
#define RUSTRT_R14   6
#define RUSTRT_R15   7
#define RUSTRT_IP    8

#define RUSTRT_MAX  32

// ARG0 is the register in which the first argument goes.
// Naturally this depends on your operating system.
#   define RUSTRT_ARG0_S r0
#   define RUSTRT_ARG1_S r1
#   define RUSTRT_ARG2_S r2
#   define RUSTRT_ARG3_S r3
