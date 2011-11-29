.text

#if defined(__APPLE__) || defined(_WIN32)
#define RECORD_SP          _record_sp
#else
#define RECORD_SP          record_sp
#endif

.globl RECORD_SP

#if defined(__linux__)
RECORD_SP:
	movq %rdi, %fs:112
	ret
#else
RECORD_SP:
	ret
#endif