#define RUSTRT_RBX   0
#define RUSTRT_RSP   1
#define RUSTRT_RBP   2
#define RUSTRT_ARG0  3 // RCX on Windows, RDI elsewhere
#define RUSTRT_R12   4
#define RUSTRT_R13   5
#define RUSTRT_R14   6
#define RUSTRT_R15   7
#define RUSTRT_IP    8
#define RUSTRT_XXX   9 // Not used, just padding
#define RUSTRT_XMM0 10
#define RUSTRT_XMM1 12
#define RUSTRT_XMM2 14
#define RUSTRT_XMM3 16
#define RUSTRT_XMM4 18
#define RUSTRT_XMM5 20
#define RUSTRT_MAX  22

// ARG0 is the register in which the first argument goes.
// Naturally this depends on your operating system.
#if defined(__MINGW32__) || defined(_WINDOWS)
#   define RUSTRT_ARG0_S %rcx
#   define RUSTRT_ARG1_S %rdx
#   define RUSTRT_ARG2_S %r8
#   define RUSTRT_ARG3_S %r9
#else
#   define RUSTRT_ARG0_S %rdi
#   define RUSTRT_ARG1_S %rsi
#   define RUSTRT_ARG2_S %rdx
#   define RUSTRT_ARG3_S %rcx
#   define RUSTRT_ARG4_S %r8
#   define RUSTRT_ARG5_S %r9
#endif


