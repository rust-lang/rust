/* markdown.h - generic markdown parser */

/*
 * Copyright (c) 2009, Natacha Porté
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

#ifndef UPSKIRT_MARKDOWN_H
#define UPSKIRT_MARKDOWN_H

#include "buffer.h"
#include "autolink.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SUNDOWN_VERSION "1.16.0"
#define SUNDOWN_VER_MAJOR 1
#define SUNDOWN_VER_MINOR 16
#define SUNDOWN_VER_REVISION 0

/********************
 * TYPE DEFINITIONS *
 ********************/

/* mkd_autolink - type of autolink */
enum mkd_autolink {
	MKDA_NOT_AUTOLINK,	/* used internally when it is not an autolink*/
	MKDA_NORMAL,		/* normal http/http/ftp/mailto/etc link */
	MKDA_EMAIL,			/* e-mail link without explit mailto: */
};

enum mkd_tableflags {
	MKD_TABLE_ALIGN_L = 1,
	MKD_TABLE_ALIGN_R = 2,
	MKD_TABLE_ALIGN_CENTER = 3,
	MKD_TABLE_ALIGNMASK = 3,
	MKD_TABLE_HEADER = 4
};

enum mkd_extensions {
	MKDEXT_NO_INTRA_EMPHASIS = (1 << 0),
	MKDEXT_TABLES = (1 << 1),
	MKDEXT_FENCED_CODE = (1 << 2),
	MKDEXT_AUTOLINK = (1 << 3),
	MKDEXT_STRIKETHROUGH = (1 << 4),
	MKDEXT_SPACE_HEADERS = (1 << 6),
	MKDEXT_SUPERSCRIPT = (1 << 7),
	MKDEXT_LAX_SPACING = (1 << 8),
};

/* sd_callbacks - functions for rendering parsed data */
struct sd_callbacks {
	/* block level callbacks - NULL skips the block */
	void (*blockcode)(struct buf *ob, const struct buf *text, const struct buf *lang, void *opaque);
	void (*blockquote)(struct buf *ob, const struct buf *text, void *opaque);
	void (*blockhtml)(struct buf *ob,const  struct buf *text, void *opaque);
	void (*header)(struct buf *ob, const struct buf *text, int level, void *opaque);
	void (*hrule)(struct buf *ob, void *opaque);
	void (*list)(struct buf *ob, const struct buf *text, int flags, void *opaque);
	void (*listitem)(struct buf *ob, const struct buf *text, int flags, void *opaque);
	void (*paragraph)(struct buf *ob, const struct buf *text, void *opaque);
	void (*table)(struct buf *ob, const struct buf *header, const struct buf *body, void *opaque);
	void (*table_row)(struct buf *ob, const struct buf *text, void *opaque);
	void (*table_cell)(struct buf *ob, const struct buf *text, int flags, void *opaque);


	/* span level callbacks - NULL or return 0 prints the span verbatim */
	int (*autolink)(struct buf *ob, const struct buf *link, enum mkd_autolink type, void *opaque);
	int (*codespan)(struct buf *ob, const struct buf *text, void *opaque);
	int (*double_emphasis)(struct buf *ob, const struct buf *text, void *opaque);
	int (*emphasis)(struct buf *ob, const struct buf *text, void *opaque);
	int (*image)(struct buf *ob, const struct buf *link, const struct buf *title, const struct buf *alt, void *opaque);
	int (*linebreak)(struct buf *ob, void *opaque);
	int (*link)(struct buf *ob, const struct buf *link, const struct buf *title, const struct buf *content, void *opaque);
	int (*raw_html_tag)(struct buf *ob, const struct buf *tag, void *opaque);
	int (*triple_emphasis)(struct buf *ob, const struct buf *text, void *opaque);
	int (*strikethrough)(struct buf *ob, const struct buf *text, void *opaque);
	int (*superscript)(struct buf *ob, const struct buf *text, void *opaque);

	/* low level callbacks - NULL copies input directly into the output */
	void (*entity)(struct buf *ob, const struct buf *entity, void *opaque);
	void (*normal_text)(struct buf *ob, const struct buf *text, void *opaque);

	/* header and footer */
	void (*doc_header)(struct buf *ob, void *opaque);
	void (*doc_footer)(struct buf *ob, void *opaque);
};

struct sd_markdown;

/*********
 * FLAGS *
 *********/

/* list/listitem flags */
#define MKD_LIST_ORDERED	1
#define MKD_LI_BLOCK		2  /* <li> containing block data */

/**********************
 * EXPORTED FUNCTIONS *
 **********************/

extern struct sd_markdown *
sd_markdown_new(
	unsigned int extensions,
	size_t max_nesting,
	const struct sd_callbacks *callbacks,
	void *opaque);

extern void
sd_markdown_render(struct buf *ob, const uint8_t *document, size_t doc_size, struct sd_markdown *md);

extern void
sd_markdown_free(struct sd_markdown *md);

extern void
sd_version(int *major, int *minor, int *revision);

#ifdef __cplusplus
}
#endif

#endif

/* vim: set filetype=c: */
