#define	assert(e) do {							\
	if (config_debug && !(e)) {					\
		malloc_write("<jemalloc>: Failed assertion\n");		\
		abort();						\
	}								\
} while (0)

#define	not_reached() do {						\
	if (config_debug) {						\
		malloc_write("<jemalloc>: Unreachable code reached\n");	\
		abort();						\
	}								\
} while (0)

#define	not_implemented() do {						\
	if (config_debug) {						\
		malloc_write("<jemalloc>: Not implemented\n");		\
		abort();						\
	}								\
} while (0)

#define	JEMALLOC_UTIL_C_
#include "jemalloc/internal/jemalloc_internal.h"

/******************************************************************************/
/* Function prototypes for non-inline static functions. */

static void	wrtmessage(void *cbopaque, const char *s);
#define	U2S_BUFSIZE	((1U << (LG_SIZEOF_INTMAX_T + 3)) + 1)
static char	*u2s(uintmax_t x, unsigned base, bool uppercase, char *s,
    size_t *slen_p);
#define	D2S_BUFSIZE	(1 + U2S_BUFSIZE)
static char	*d2s(intmax_t x, char sign, char *s, size_t *slen_p);
#define	O2S_BUFSIZE	(1 + U2S_BUFSIZE)
static char	*o2s(uintmax_t x, bool alt_form, char *s, size_t *slen_p);
#define	X2S_BUFSIZE	(2 + U2S_BUFSIZE)
static char	*x2s(uintmax_t x, bool alt_form, bool uppercase, char *s,
    size_t *slen_p);

/******************************************************************************/

/* malloc_message() setup. */
static void
wrtmessage(void *cbopaque, const char *s)
{

#ifdef SYS_write
	/*
	 * Use syscall(2) rather than write(2) when possible in order to avoid
	 * the possibility of memory allocation within libc.  This is necessary
	 * on FreeBSD; most operating systems do not have this problem though.
	 */
	UNUSED int result = syscall(SYS_write, STDERR_FILENO, s, strlen(s));
#else
	UNUSED int result = write(STDERR_FILENO, s, strlen(s));
#endif
}

JEMALLOC_EXPORT void	(*je_malloc_message)(void *, const char *s);

/*
 * Wrapper around malloc_message() that avoids the need for
 * je_malloc_message(...) throughout the code.
 */
void
malloc_write(const char *s)
{

	if (je_malloc_message != NULL)
		je_malloc_message(NULL, s);
	else
		wrtmessage(NULL, s);
}

/*
 * glibc provides a non-standard strerror_r() when _GNU_SOURCE is defined, so
 * provide a wrapper.
 */
int
buferror(char *buf, size_t buflen)
{

#ifdef _WIN32
	FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), 0,
	    (LPSTR)buf, buflen, NULL);
	return (0);
#elif defined(_GNU_SOURCE)
	char *b = strerror_r(errno, buf, buflen);
	if (b != buf) {
		strncpy(buf, b, buflen);
		buf[buflen-1] = '\0';
	}
	return (0);
#else
	return (strerror_r(errno, buf, buflen));
#endif
}

uintmax_t
malloc_strtoumax(const char *nptr, char **endptr, int base)
{
	uintmax_t ret, digit;
	int b;
	bool neg;
	const char *p, *ns;

	if (base < 0 || base == 1 || base > 36) {
		set_errno(EINVAL);
		return (UINTMAX_MAX);
	}
	b = base;

	/* Swallow leading whitespace and get sign, if any. */
	neg = false;
	p = nptr;
	while (true) {
		switch (*p) {
		case '\t': case '\n': case '\v': case '\f': case '\r': case ' ':
			p++;
			break;
		case '-':
			neg = true;
			/* Fall through. */
		case '+':
			p++;
			/* Fall through. */
		default:
			goto label_prefix;
		}
	}

	/* Get prefix, if any. */
	label_prefix:
	/*
	 * Note where the first non-whitespace/sign character is so that it is
	 * possible to tell whether any digits are consumed (e.g., "  0" vs.
	 * "  -x").
	 */
	ns = p;
	if (*p == '0') {
		switch (p[1]) {
		case '0': case '1': case '2': case '3': case '4': case '5':
		case '6': case '7':
			if (b == 0)
				b = 8;
			if (b == 8)
				p++;
			break;
		case 'x':
			switch (p[2]) {
			case '0': case '1': case '2': case '3': case '4':
			case '5': case '6': case '7': case '8': case '9':
			case 'A': case 'B': case 'C': case 'D': case 'E':
			case 'F':
			case 'a': case 'b': case 'c': case 'd': case 'e':
			case 'f':
				if (b == 0)
					b = 16;
				if (b == 16)
					p += 2;
				break;
			default:
				break;
			}
			break;
		default:
			break;
		}
	}
	if (b == 0)
		b = 10;

	/* Convert. */
	ret = 0;
	while ((*p >= '0' && *p <= '9' && (digit = *p - '0') < b)
	    || (*p >= 'A' && *p <= 'Z' && (digit = 10 + *p - 'A') < b)
	    || (*p >= 'a' && *p <= 'z' && (digit = 10 + *p - 'a') < b)) {
		uintmax_t pret = ret;
		ret *= b;
		ret += digit;
		if (ret < pret) {
			/* Overflow. */
			set_errno(ERANGE);
			return (UINTMAX_MAX);
		}
		p++;
	}
	if (neg)
		ret = -ret;

	if (endptr != NULL) {
		if (p == ns) {
			/* No characters were converted. */
			*endptr = (char *)nptr;
		} else
			*endptr = (char *)p;
	}

	return (ret);
}

static char *
u2s(uintmax_t x, unsigned base, bool uppercase, char *s, size_t *slen_p)
{
	unsigned i;

	i = U2S_BUFSIZE - 1;
	s[i] = '\0';
	switch (base) {
	case 10:
		do {
			i--;
			s[i] = "0123456789"[x % (uint64_t)10];
			x /= (uint64_t)10;
		} while (x > 0);
		break;
	case 16: {
		const char *digits = (uppercase)
		    ? "0123456789ABCDEF"
		    : "0123456789abcdef";

		do {
			i--;
			s[i] = digits[x & 0xf];
			x >>= 4;
		} while (x > 0);
		break;
	} default: {
		const char *digits = (uppercase)
		    ? "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		    : "0123456789abcdefghijklmnopqrstuvwxyz";

		assert(base >= 2 && base <= 36);
		do {
			i--;
			s[i] = digits[x % (uint64_t)base];
			x /= (uint64_t)base;
		} while (x > 0);
	}}

	*slen_p = U2S_BUFSIZE - 1 - i;
	return (&s[i]);
}

static char *
d2s(intmax_t x, char sign, char *s, size_t *slen_p)
{
	bool neg;

	if ((neg = (x < 0)))
		x = -x;
	s = u2s(x, 10, false, s, slen_p);
	if (neg)
		sign = '-';
	switch (sign) {
	case '-':
		if (neg == false)
			break;
		/* Fall through. */
	case ' ':
	case '+':
		s--;
		(*slen_p)++;
		*s = sign;
		break;
	default: not_reached();
	}
	return (s);
}

static char *
o2s(uintmax_t x, bool alt_form, char *s, size_t *slen_p)
{

	s = u2s(x, 8, false, s, slen_p);
	if (alt_form && *s != '0') {
		s--;
		(*slen_p)++;
		*s = '0';
	}
	return (s);
}

static char *
x2s(uintmax_t x, bool alt_form, bool uppercase, char *s, size_t *slen_p)
{

	s = u2s(x, 16, uppercase, s, slen_p);
	if (alt_form) {
		s -= 2;
		(*slen_p) += 2;
		memcpy(s, uppercase ? "0X" : "0x", 2);
	}
	return (s);
}

int
malloc_vsnprintf(char *str, size_t size, const char *format, va_list ap)
{
	int ret;
	size_t i;
	const char *f;

#define	APPEND_C(c) do {						\
	if (i < size)							\
		str[i] = (c);						\
	i++;								\
} while (0)
#define	APPEND_S(s, slen) do {						\
	if (i < size) {							\
		size_t cpylen = (slen <= size - i) ? slen : size - i;	\
		memcpy(&str[i], s, cpylen);				\
	}								\
	i += slen;							\
} while (0)
#define	APPEND_PADDED_S(s, slen, width, left_justify) do {		\
	/* Left padding. */						\
	size_t pad_len = (width == -1) ? 0 : ((slen < (size_t)width) ?	\
	    (size_t)width - slen : 0);					\
	if (left_justify == false && pad_len != 0) {			\
		size_t j;						\
		for (j = 0; j < pad_len; j++)				\
			APPEND_C(' ');					\
	}								\
	/* Value. */							\
	APPEND_S(s, slen);						\
	/* Right padding. */						\
	if (left_justify && pad_len != 0) {				\
		size_t j;						\
		for (j = 0; j < pad_len; j++)				\
			APPEND_C(' ');					\
	}								\
} while (0)
#define GET_ARG_NUMERIC(val, len) do {					\
	switch (len) {							\
	case '?':							\
		val = va_arg(ap, int);					\
		break;							\
	case '?' | 0x80:						\
		val = va_arg(ap, unsigned int);				\
		break;							\
	case 'l':							\
		val = va_arg(ap, long);					\
		break;							\
	case 'l' | 0x80:						\
		val = va_arg(ap, unsigned long);			\
		break;							\
	case 'q':							\
		val = va_arg(ap, long long);				\
		break;							\
	case 'q' | 0x80:						\
		val = va_arg(ap, unsigned long long);			\
		break;							\
	case 'j':							\
		val = va_arg(ap, intmax_t);				\
		break;							\
	case 't':							\
		val = va_arg(ap, ptrdiff_t);				\
		break;							\
	case 'z':							\
		val = va_arg(ap, ssize_t);				\
		break;							\
	case 'z' | 0x80:						\
		val = va_arg(ap, size_t);				\
		break;							\
	case 'p': /* Synthetic; used for %p. */				\
		val = va_arg(ap, uintptr_t);				\
		break;							\
	default: not_reached();						\
	}								\
} while (0)

	i = 0;
	f = format;
	while (true) {
		switch (*f) {
		case '\0': goto label_out;
		case '%': {
			bool alt_form = false;
			bool left_justify = false;
			bool plus_space = false;
			bool plus_plus = false;
			int prec = -1;
			int width = -1;
			unsigned char len = '?';

			f++;
			if (*f == '%') {
				/* %% */
				APPEND_C(*f);
				break;
			}
			/* Flags. */
			while (true) {
				switch (*f) {
				case '#':
					assert(alt_form == false);
					alt_form = true;
					break;
				case '-':
					assert(left_justify == false);
					left_justify = true;
					break;
				case ' ':
					assert(plus_space == false);
					plus_space = true;
					break;
				case '+':
					assert(plus_plus == false);
					plus_plus = true;
					break;
				default: goto label_width;
				}
				f++;
			}
			/* Width. */
			label_width:
			switch (*f) {
			case '*':
				width = va_arg(ap, int);
				f++;
				break;
			case '0': case '1': case '2': case '3': case '4':
			case '5': case '6': case '7': case '8': case '9': {
				uintmax_t uwidth;
				set_errno(0);
				uwidth = malloc_strtoumax(f, (char **)&f, 10);
				assert(uwidth != UINTMAX_MAX || get_errno() !=
				    ERANGE);
				width = (int)uwidth;
				if (*f == '.') {
					f++;
					goto label_precision;
				} else
					goto label_length;
				break;
			} case '.':
				f++;
				goto label_precision;
			default: goto label_length;
			}
			/* Precision. */
			label_precision:
			switch (*f) {
			case '*':
				prec = va_arg(ap, int);
				f++;
				break;
			case '0': case '1': case '2': case '3': case '4':
			case '5': case '6': case '7': case '8': case '9': {
				uintmax_t uprec;
				set_errno(0);
				uprec = malloc_strtoumax(f, (char **)&f, 10);
				assert(uprec != UINTMAX_MAX || get_errno() !=
				    ERANGE);
				prec = (int)uprec;
				break;
			}
			default: break;
			}
			/* Length. */
			label_length:
			switch (*f) {
			case 'l':
				f++;
				if (*f == 'l') {
					len = 'q';
					f++;
				} else
					len = 'l';
				break;
			case 'j':
				len = 'j';
				f++;
				break;
			case 't':
				len = 't';
				f++;
				break;
			case 'z':
				len = 'z';
				f++;
				break;
			default: break;
			}
			/* Conversion specifier. */
			switch (*f) {
				char *s;
				size_t slen;
			case 'd': case 'i': {
				intmax_t val JEMALLOC_CC_SILENCE_INIT(0);
				char buf[D2S_BUFSIZE];

				GET_ARG_NUMERIC(val, len);
				s = d2s(val, (plus_plus ? '+' : (plus_space ?
				    ' ' : '-')), buf, &slen);
				APPEND_PADDED_S(s, slen, width, left_justify);
				f++;
				break;
			} case 'o': {
				uintmax_t val JEMALLOC_CC_SILENCE_INIT(0);
				char buf[O2S_BUFSIZE];

				GET_ARG_NUMERIC(val, len | 0x80);
				s = o2s(val, alt_form, buf, &slen);
				APPEND_PADDED_S(s, slen, width, left_justify);
				f++;
				break;
			} case 'u': {
				uintmax_t val JEMALLOC_CC_SILENCE_INIT(0);
				char buf[U2S_BUFSIZE];

				GET_ARG_NUMERIC(val, len | 0x80);
				s = u2s(val, 10, false, buf, &slen);
				APPEND_PADDED_S(s, slen, width, left_justify);
				f++;
				break;
			} case 'x': case 'X': {
				uintmax_t val JEMALLOC_CC_SILENCE_INIT(0);
				char buf[X2S_BUFSIZE];

				GET_ARG_NUMERIC(val, len | 0x80);
				s = x2s(val, alt_form, *f == 'X', buf, &slen);
				APPEND_PADDED_S(s, slen, width, left_justify);
				f++;
				break;
			} case 'c': {
				unsigned char val;
				char buf[2];

				assert(len == '?' || len == 'l');
				assert_not_implemented(len != 'l');
				val = va_arg(ap, int);
				buf[0] = val;
				buf[1] = '\0';
				APPEND_PADDED_S(buf, 1, width, left_justify);
				f++;
				break;
			} case 's':
				assert(len == '?' || len == 'l');
				assert_not_implemented(len != 'l');
				s = va_arg(ap, char *);
				slen = (prec == -1) ? strlen(s) : prec;
				APPEND_PADDED_S(s, slen, width, left_justify);
				f++;
				break;
			case 'p': {
				uintmax_t val;
				char buf[X2S_BUFSIZE];

				GET_ARG_NUMERIC(val, 'p');
				s = x2s(val, true, false, buf, &slen);
				APPEND_PADDED_S(s, slen, width, left_justify);
				f++;
				break;
			}
			default: not_implemented();
			}
			break;
		} default: {
			APPEND_C(*f);
			f++;
			break;
		}}
	}
	label_out:
	if (i < size)
		str[i] = '\0';
	else
		str[size - 1] = '\0';
	ret = i;

#undef APPEND_C
#undef APPEND_S
#undef APPEND_PADDED_S
#undef GET_ARG_NUMERIC
	return (ret);
}

JEMALLOC_ATTR(format(printf, 3, 4))
int
malloc_snprintf(char *str, size_t size, const char *format, ...)
{
	int ret;
	va_list ap;

	va_start(ap, format);
	ret = malloc_vsnprintf(str, size, format, ap);
	va_end(ap);

	return (ret);
}

void
malloc_vcprintf(void (*write_cb)(void *, const char *), void *cbopaque,
    const char *format, va_list ap)
{
	char buf[MALLOC_PRINTF_BUFSIZE];

	if (write_cb == NULL) {
		/*
		 * The caller did not provide an alternate write_cb callback
		 * function, so use the default one.  malloc_write() is an
		 * inline function, so use malloc_message() directly here.
		 */
		write_cb = (je_malloc_message != NULL) ? je_malloc_message :
		    wrtmessage;
		cbopaque = NULL;
	}

	malloc_vsnprintf(buf, sizeof(buf), format, ap);
	write_cb(cbopaque, buf);
}

/*
 * Print to a callback function in such a way as to (hopefully) avoid memory
 * allocation.
 */
JEMALLOC_ATTR(format(printf, 3, 4))
void
malloc_cprintf(void (*write_cb)(void *, const char *), void *cbopaque,
    const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	malloc_vcprintf(write_cb, cbopaque, format, ap);
	va_end(ap);
}

/* Print to stderr in such a way as to avoid memory allocation. */
JEMALLOC_ATTR(format(printf, 1, 2))
void
malloc_printf(const char *format, ...)
{
	va_list ap;

	va_start(ap, format);
	malloc_vcprintf(NULL, NULL, format, ap);
	va_end(ap);
}
