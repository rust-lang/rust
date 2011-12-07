.text

#if defined(__APPLE__) || defined(_WIN32)
#define RECORD_SP          _record_sp
#define GET_SP             _get_sp
#else
#define RECORD_SP          record_sp
#define GET_SP             get_sp
#endif

.globl RECORD_SP
.globl GET_SP

#if defined(__linux__)
RECORD_SP:
	movq %rdi, %fs:112
	ret
#else
#if defined(__APPLE__)
RECORD_SP:
	movq $0x60+90*8, %rsi
	movq %rdi, %gs:(%rsi)
	ret
#else
RECORD_SP:
	ret
#endif
#endif

GET_SP:
	movq %rsp, %rax
	ret
