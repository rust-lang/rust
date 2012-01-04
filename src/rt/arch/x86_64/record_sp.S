.text

#if defined(__APPLE__) || defined(_WIN32)
#define RECORD_SP          _record_sp
#define GET_SP             _get_sp
#define CHECK_STACK        _check_stack_alignment
#else
#define RECORD_SP          record_sp
#define GET_SP             get_sp
#define CHECK_STACK        check_stack_alignment
#endif

.globl RECORD_SP
.globl GET_SP
.globl CHECK_STACK

#if defined(__linux__)
RECORD_SP:
	movq %rdi, %fs:112
	ret
#elif defined(__APPLE__)
RECORD_SP:
	movq $0x60+90*8, %rsi
	movq %rdi, %gs:(%rsi)
	ret
#elif defined(__FreeBSD__)
RECORD_SP:
	movq %rdi, %fs:24
	ret
#else
RECORD_SP:
	ret
#endif

GET_SP:
	movq %rsp, %rax
	ret

// This will segfault if not called on a 16-byte boundary
CHECK_STACK:
	subq $24, %rsp
	movaps %xmm0, (%rsp)
	addq $24, %rsp
	ret
