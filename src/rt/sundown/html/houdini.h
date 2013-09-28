#ifndef HOUDINI_H__
#define HOUDINI_H__

#include "buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HOUDINI_USE_LOCALE
#	define _isxdigit(c) isxdigit(c)
#	define _isdigit(c) isdigit(c)
#else
/*
 * Helper _isdigit methods -- do not trust the current locale
 * */
#	define _isxdigit(c) (strchr("0123456789ABCDEFabcdef", (c)) != NULL)
#	define _isdigit(c) ((c) >= '0' && (c) <= '9')
#endif

extern void houdini_escape_html(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_escape_html0(struct buf *ob, const uint8_t *src, size_t size, int secure);
extern void houdini_unescape_html(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_escape_xml(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_escape_uri(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_escape_url(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_escape_href(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_unescape_uri(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_unescape_url(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_escape_js(struct buf *ob, const uint8_t *src, size_t size);
extern void houdini_unescape_js(struct buf *ob, const uint8_t *src, size_t size);

#ifdef __cplusplus
}
#endif

#endif
