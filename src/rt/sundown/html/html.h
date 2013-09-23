/*
 * Copyright (c) 2011, Vicent Marti
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

#ifndef UPSKIRT_HTML_H
#define UPSKIRT_HTML_H

#include "markdown.h"
#include "buffer.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct html_renderopt {
	struct {
		int header_count;
		int current_level;
		int level_offset;
	} toc_data;

	unsigned int flags;

	/* extra callbacks */
	void (*link_attributes)(struct buf *ob, const struct buf *url, void *self);
};

typedef enum {
	HTML_SKIP_HTML = (1 << 0),
	HTML_SKIP_STYLE = (1 << 1),
	HTML_SKIP_IMAGES = (1 << 2),
	HTML_SKIP_LINKS = (1 << 3),
	HTML_EXPAND_TABS = (1 << 4),
	HTML_SAFELINK = (1 << 5),
	HTML_TOC = (1 << 6),
	HTML_HARD_WRAP = (1 << 7),
	HTML_USE_XHTML = (1 << 8),
	HTML_ESCAPE = (1 << 9),
} html_render_mode;

typedef enum {
	HTML_TAG_NONE = 0,
	HTML_TAG_OPEN,
	HTML_TAG_CLOSE,
} html_tag;

int
sdhtml_is_tag(const uint8_t *tag_data, size_t tag_size, const char *tagname);

extern void
sdhtml_renderer(struct sd_callbacks *callbacks, struct html_renderopt *options_ptr, unsigned int render_flags);

extern void
sdhtml_toc_renderer(struct sd_callbacks *callbacks, struct html_renderopt *options_ptr);

extern void
sdhtml_smartypants(struct buf *ob, const uint8_t *text, size_t size);

#ifdef __cplusplus
}
#endif

#endif

