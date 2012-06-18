#include "rust_type.h"
#include "rust_shape.h"


// A hardcoded type descriptor for strings, since the runtime needs to
// be able to create them.

struct rust_shape_tables empty_shape_tables;

uint8_t str_body_shape[] = {
    shape::SHAPE_UNBOXED_VEC,
    0x1, // is_pod
    0x1, 0x0, // size field: 1
    shape::SHAPE_U8
};

struct type_desc str_body_tydesc = {
    0, // unused
    1, // size
    1, // align
    NULL, // take_glue
    NULL, // drop_glue
    NULL, // free_glue
    NULL, // visit_glue
    0, // unused
    0, // unused
    0, // unused
    0, // unused
    str_body_shape, // shape
    &empty_shape_tables, // shape_tables
    0, // unused
    0, // unused
};

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
