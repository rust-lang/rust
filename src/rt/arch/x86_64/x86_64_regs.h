#define RUSTRT_RBX  0
#define RUSTRT_RSP  1
#define RUSTRT_RBP  2
#define RUSTRT_RDI  3
#define RUSTRT_RCX  4
#define RUSTRT_R12  5
#define RUSTRT_R13  6
#define RUSTRT_R14  7
#define RUSTRT_R15  8
#define RUSTRT_IP   9
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
#   define RUSTRT_ARG0 RUSTRT_RCX  
#   define RUSTRT_ARG0_S %rcx
#   define RUSTRT_ARG1_S %rdx
#else
#   define RUSTRT_ARG0 RUSTRT_RDI  
#endif


