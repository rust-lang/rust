.text

#if defined(__APPLE__) || defined(_WIN32)
#define RECORD_SP          _record_sp
#else
#define RECORD_SP          record_sp
#endif

.globl RECORD_SP

RECORD_SP:
	ret
