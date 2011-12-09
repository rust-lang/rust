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
	movl 4(%esp), %eax
	movl %eax, %gs:48
	ret
#endif

#if defined(__APPLE__)
RECORD_SP:
	movl $0x48+90*4, %eax
	movl 4(%esp), %ecx
	movl %ecx, %gs:(%eax)
	ret
#endif

#if defined(_WIN32)
RECORD_SP:
	movl 4(%esp), %eax
	movl %eax, %fs:0x14
	ret
#endif

GET_SP:
	movl %esp, %eax
	ret