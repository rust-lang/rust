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

#include "buffer.h"
#include "html.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#if defined(_WIN32)
#define snprintf	_snprintf		
#endif

struct smartypants_data {
	int in_squote;
	int in_dquote;
};

static size_t smartypants_cb__ltag(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__dquote(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__amp(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__period(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__number(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__dash(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__parens(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__squote(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__backtick(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);
static size_t smartypants_cb__escape(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size);

static size_t (*smartypants_cb_ptrs[])
	(struct buf *, struct smartypants_data *, uint8_t, const uint8_t *, size_t) =
{
	NULL,					/* 0 */
	smartypants_cb__dash,	/* 1 */
	smartypants_cb__parens,	/* 2 */
	smartypants_cb__squote, /* 3 */
	smartypants_cb__dquote, /* 4 */
	smartypants_cb__amp,	/* 5 */
	smartypants_cb__period,	/* 6 */
	smartypants_cb__number,	/* 7 */
	smartypants_cb__ltag,	/* 8 */
	smartypants_cb__backtick, /* 9 */
	smartypants_cb__escape, /* 10 */
};

static const uint8_t smartypants_cb_chars[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 4, 0, 0, 0, 5, 3, 2, 0, 0, 0, 0, 1, 6, 0,
	0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0,
	9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static inline int
word_boundary(uint8_t c)
{
	return c == 0 || isspace(c) || ispunct(c);
}

static int
smartypants_quotes(struct buf *ob, uint8_t previous_char, uint8_t next_char, uint8_t quote, int *is_open)
{
	char ent[8];

	if (*is_open && !word_boundary(next_char))
		return 0;

	if (!(*is_open) && !word_boundary(previous_char))
		return 0;

	snprintf(ent, sizeof(ent), "&%c%cquo;", (*is_open) ? 'r' : 'l', quote);
	*is_open = !(*is_open);
	bufputs(ob, ent);
	return 1;
}

static size_t
smartypants_cb__squote(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size >= 2) {
		uint8_t t1 = tolower(text[1]);

		if (t1 == '\'') {
			if (smartypants_quotes(ob, previous_char, size >= 3 ? text[2] : 0, 'd', &smrt->in_dquote))
				return 1;
		}

		if ((t1 == 's' || t1 == 't' || t1 == 'm' || t1 == 'd') &&
			(size == 3 || word_boundary(text[2]))) {
			BUFPUTSL(ob, "&rsquo;");
			return 0;
		}

		if (size >= 3) {
			uint8_t t2 = tolower(text[2]);

			if (((t1 == 'r' && t2 == 'e') ||
				(t1 == 'l' && t2 == 'l') ||
				(t1 == 'v' && t2 == 'e')) &&
				(size == 4 || word_boundary(text[3]))) {
				BUFPUTSL(ob, "&rsquo;");
				return 0;
			}
		}
	}

	if (smartypants_quotes(ob, previous_char, size > 0 ? text[1] : 0, 's', &smrt->in_squote))
		return 0;

	bufputc(ob, text[0]);
	return 0;
}

static size_t
smartypants_cb__parens(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size >= 3) {
		uint8_t t1 = tolower(text[1]);
		uint8_t t2 = tolower(text[2]);

		if (t1 == 'c' && t2 == ')') {
			BUFPUTSL(ob, "&copy;");
			return 2;
		}

		if (t1 == 'r' && t2 == ')') {
			BUFPUTSL(ob, "&reg;");
			return 2;
		}

		if (size >= 4 && t1 == 't' && t2 == 'm' && text[3] == ')') {
			BUFPUTSL(ob, "&trade;");
			return 3;
		}
	}

	bufputc(ob, text[0]);
	return 0;
}

static size_t
smartypants_cb__dash(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size >= 3 && text[1] == '-' && text[2] == '-') {
		BUFPUTSL(ob, "&mdash;");
		return 2;
	}

	if (size >= 2 && text[1] == '-') {
		BUFPUTSL(ob, "&ndash;");
		return 1;
	}

	bufputc(ob, text[0]);
	return 0;
}

static size_t
smartypants_cb__amp(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size >= 6 && memcmp(text, "&quot;", 6) == 0) {
		if (smartypants_quotes(ob, previous_char, size >= 7 ? text[6] : 0, 'd', &smrt->in_dquote))
			return 5;
	}

	if (size >= 4 && memcmp(text, "&#0;", 4) == 0)
		return 3;

	bufputc(ob, '&');
	return 0;
}

static size_t
smartypants_cb__period(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size >= 3 && text[1] == '.' && text[2] == '.') {
		BUFPUTSL(ob, "&hellip;");
		return 2;
	}

	if (size >= 5 && text[1] == ' ' && text[2] == '.' && text[3] == ' ' && text[4] == '.') {
		BUFPUTSL(ob, "&hellip;");
		return 4;
	}

	bufputc(ob, text[0]);
	return 0;
}

static size_t
smartypants_cb__backtick(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size >= 2 && text[1] == '`') {
		if (smartypants_quotes(ob, previous_char, size >= 3 ? text[2] : 0, 'd', &smrt->in_dquote))
			return 1;
	}

	return 0;
}

static size_t
smartypants_cb__number(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (word_boundary(previous_char) && size >= 3) {
		if (text[0] == '1' && text[1] == '/' && text[2] == '2') {
			if (size == 3 || word_boundary(text[3])) {
				BUFPUTSL(ob, "&frac12;");
				return 2;
			}
		}

		if (text[0] == '1' && text[1] == '/' && text[2] == '4') {
			if (size == 3 || word_boundary(text[3]) ||
				(size >= 5 && tolower(text[3]) == 't' && tolower(text[4]) == 'h')) {
				BUFPUTSL(ob, "&frac14;");
				return 2;
			}
		}

		if (text[0] == '3' && text[1] == '/' && text[2] == '4') {
			if (size == 3 || word_boundary(text[3]) ||
				(size >= 6 && tolower(text[3]) == 't' && tolower(text[4]) == 'h' && tolower(text[5]) == 's')) {
				BUFPUTSL(ob, "&frac34;");
				return 2;
			}
		}
	}

	bufputc(ob, text[0]);
	return 0;
}

static size_t
smartypants_cb__dquote(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (!smartypants_quotes(ob, previous_char, size > 0 ? text[1] : 0, 'd', &smrt->in_dquote))
		BUFPUTSL(ob, "&quot;");

	return 0;
}

static size_t
smartypants_cb__ltag(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	static const char *skip_tags[] = {
	  "pre", "code", "var", "samp", "kbd", "math", "script", "style"
	};
	static const size_t skip_tags_count = 8;

	size_t tag, i = 0;

	while (i < size && text[i] != '>')
		i++;

	for (tag = 0; tag < skip_tags_count; ++tag) {
		if (sdhtml_is_tag(text, size, skip_tags[tag]) == HTML_TAG_OPEN)
			break;
	}

	if (tag < skip_tags_count) {
		for (;;) {
			while (i < size && text[i] != '<')
				i++;

			if (i == size)
				break;

			if (sdhtml_is_tag(text + i, size - i, skip_tags[tag]) == HTML_TAG_CLOSE)
				break;

			i++;
		}

		while (i < size && text[i] != '>')
			i++;
	}

	bufput(ob, text, i + 1);
	return i;
}

static size_t
smartypants_cb__escape(struct buf *ob, struct smartypants_data *smrt, uint8_t previous_char, const uint8_t *text, size_t size)
{
	if (size < 2)
		return 0;

	switch (text[1]) {
	case '\\':
	case '"':
	case '\'':
	case '.':
	case '-':
	case '`':
		bufputc(ob, text[1]);
		return 1;

	default:
		bufputc(ob, '\\');
		return 0;
	}
}

#if 0
static struct {
    uint8_t c0;
    const uint8_t *pattern;
    const uint8_t *entity;
    int skip;
} smartypants_subs[] = {
    { '\'', "'s>",      "&rsquo;",  0 },
    { '\'', "'t>",      "&rsquo;",  0 },
    { '\'', "'re>",     "&rsquo;",  0 },
    { '\'', "'ll>",     "&rsquo;",  0 },
    { '\'', "'ve>",     "&rsquo;",  0 },
    { '\'', "'m>",      "&rsquo;",  0 },
    { '\'', "'d>",      "&rsquo;",  0 },
    { '-',  "--",       "&mdash;",  1 },
    { '-',  "<->",      "&ndash;",  0 },
    { '.',  "...",      "&hellip;", 2 },
    { '.',  ". . .",    "&hellip;", 4 },
    { '(',  "(c)",      "&copy;",   2 },
    { '(',  "(r)",      "&reg;",    2 },
    { '(',  "(tm)",     "&trade;",  3 },
    { '3',  "<3/4>",    "&frac34;", 2 },
    { '3',  "<3/4ths>", "&frac34;", 2 },
    { '1',  "<1/2>",    "&frac12;", 2 },
    { '1',  "<1/4>",    "&frac14;", 2 },
    { '1',  "<1/4th>",  "&frac14;", 2 },
    { '&',  "&#0;",      0,       3 },
};
#endif

void
sdhtml_smartypants(struct buf *ob, const uint8_t *text, size_t size)
{
	size_t i;
	struct smartypants_data smrt = {0, 0};

	if (!text)
		return;

	bufgrow(ob, size);

	for (i = 0; i < size; ++i) {
		size_t org;
		uint8_t action = 0;

		org = i;
		while (i < size && (action = smartypants_cb_chars[text[i]]) == 0)
			i++;

		if (i > org)
			bufput(ob, text + org, i - org);

		if (i < size) {
			i += smartypants_cb_ptrs[(int)action]
				(ob, &smrt, i ? text[i - 1] : 0, text + i, size - i);
		}
	}
}


