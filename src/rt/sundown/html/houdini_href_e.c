#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "houdini.h"

#define ESCAPE_GROW_FACTOR(x) (((x) * 12) / 10)

/*
 * The following characters will not be escaped:
 *
 *		-_.+!*'(),%#@?=;:/,+&$ alphanum
 *
 * Note that this character set is the addition of:
 *
 *	- The characters which are safe to be in an URL
 *	- The characters which are *not* safe to be in
 *	an URL because they are RESERVED characters.
 *
 * We asume (lazily) that any RESERVED char that
 * appears inside an URL is actually meant to
 * have its native function (i.e. as an URL 
 * component/separator) and hence needs no escaping.
 *
 * There are two exceptions: the chacters & (amp)
 * and ' (single quote) do not appear in the table.
 * They are meant to appear in the URL as components,
 * yet they require special HTML-entity escaping
 * to generate valid HTML markup.
 *
 * All other characters will be escaped to %XX.
 *
 */
static const char HREF_SAFE[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

void
houdini_escape_href(struct buf *ob, const uint8_t *src, size_t size)
{
	static const char hex_chars[] = "0123456789ABCDEF";
	size_t  i = 0, org;
	char hex_str[3];

	bufgrow(ob, ESCAPE_GROW_FACTOR(size));
	hex_str[0] = '%';

	while (i < size) {
		org = i;
		while (i < size && HREF_SAFE[src[i]] != 0)
			i++;

		if (i > org)
			bufput(ob, src + org, i - org);

		/* escaping */
		if (i >= size)
			break;

		switch (src[i]) {
		/* amp appears all the time in URLs, but needs
		 * HTML-entity escaping to be inside an href */
		case '&': 
			BUFPUTSL(ob, "&amp;");
			break;

		/* the single quote is a valid URL character
		 * according to the standard; it needs HTML
		 * entity escaping too */
		case '\'':
			BUFPUTSL(ob, "&#x27;");
			break;
		
		/* the space can be escaped to %20 or a plus
		 * sign. we're going with the generic escape
		 * for now. the plus thing is more commonly seen
		 * when building GET strings */
#if 0
		case ' ':
			bufputc(ob, '+');
			break;
#endif

		/* every other character goes with a %XX escaping */
		default:
			hex_str[1] = hex_chars[(src[i] >> 4) & 0xF];
			hex_str[2] = hex_chars[src[i] & 0xF];
			bufput(ob, hex_str, 3);
		}

		i++;
	}
}
