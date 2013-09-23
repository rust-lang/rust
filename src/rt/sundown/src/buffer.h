/*
 * Copyright (c) 2008, Natacha Porté
 * Copyright (c) 2011, Vicent Martí
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef BUFFER_H__
#define BUFFER_H__

#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define __attribute__(x)
#define inline
#endif

typedef enum {
	BUF_OK = 0,
	BUF_ENOMEM = -1,
} buferror_t;

/* struct buf: character array buffer */
struct buf {
	uint8_t *data;		/* actual character data */
	size_t size;	/* size of the string */
	size_t asize;	/* allocated size (0 = volatile buffer) */
	size_t unit;	/* reallocation unit size (0 = read-only buffer) */
};

/* CONST_BUF: global buffer from a string litteral */
#define BUF_STATIC(string) \
	{ (uint8_t *)string, sizeof string -1, sizeof string, 0, 0 }

/* VOLATILE_BUF: macro for creating a volatile buffer on the stack */
#define BUF_VOLATILE(strname) \
	{ (uint8_t *)strname, strlen(strname), 0, 0, 0 }

/* BUFPUTSL: optimized bufputs of a string litteral */
#define BUFPUTSL(output, literal) \
	bufput(output, literal, sizeof literal - 1)

/* bufgrow: increasing the allocated size to the given value */
int bufgrow(struct buf *, size_t);

/* bufnew: allocation of a new buffer */
struct buf *bufnew(size_t) __attribute__ ((malloc));

/* bufnullterm: NUL-termination of the string array (making a C-string) */
const char *bufcstr(struct buf *);

/* bufprefix: compare the beginning of a buffer with a string */
int bufprefix(const struct buf *buf, const char *prefix);

/* bufput: appends raw data to a buffer */
void bufput(struct buf *, const void *, size_t);

/* bufputs: appends a NUL-terminated string to a buffer */
void bufputs(struct buf *, const char *);

/* bufputc: appends a single char to a buffer */
void bufputc(struct buf *, int);

/* bufrelease: decrease the reference count and free the buffer if needed */
void bufrelease(struct buf *);

/* bufreset: frees internal data of the buffer */
void bufreset(struct buf *);

/* bufslurp: removes a given number of bytes from the head of the array */
void bufslurp(struct buf *, size_t);

/* bufprintf: formatted printing to a buffer */
void bufprintf(struct buf *, const char *, ...) __attribute__ ((format (printf, 2, 3)));

#ifdef __cplusplus
}
#endif

#endif
