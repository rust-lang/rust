.text

#if defined(__APPLE__) || defined(_WIN32)
#define RECORD_SP_LIMIT    _record_sp_limit
#define GET_SP_LIMIT       _get_sp_limit
#define GET_SP             _get_sp
#else
#define RECORD_SP_LIMIT    record_sp_limit
#define GET_SP_LIMIT       get_sp_limit
#define GET_SP             get_sp
#endif

.globl RECORD_SP_LIMIT
.globl GET_SP_LIMIT
.globl GET_SP

#if defined(__linux__)
RECORD_SP_LIMIT:
	movq %rdi, %fs:112
	ret
#elif defined(__APPLE__)
RECORD_SP_LIMIT:
	movq $0x60+90*8, %rsi
	movq %rdi, %gs:(%rsi)
	ret
#elif defined(__FreeBSD__)
RECORD_SP_LIMIT:
	movq %rdi, %fs:24
	ret
#else
RECORD_SP_LIMIT:
	ret
#endif

#if defined(__linux__)
GET_SP_LIMIT:
	movq %fs:112, %rax
	ret
#elif defined(__APPLE__)
GET_SP_LIMIT:
	movq $0x60+90*8, %rsi
	movq %gs:(%rsi), %rax
	ret
#elif defined(__FreeBSD__)
GET_SP_LIMIT:
	movq %fs:24, %rax
	ret
#endif

GET_SP:
	movq %rsp, %rax
	ret
