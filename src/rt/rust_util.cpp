#include "rust_type.h"


// A hardcoded type descriptor for strings, since the runtime needs to
// be able to create them.

struct type_desc str_body_tydesc = {
    1, // size
    1, // align
    NULL, // take_glue
    NULL, // drop_glue
    NULL, // free_glue
    NULL, // visit_glue
    NULL, // shape
    NULL, // shape_tables
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
