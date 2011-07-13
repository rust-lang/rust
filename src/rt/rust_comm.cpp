
#include "rust_internal.h"

template class ptr_vec<rust_alarm>;

rust_alarm::rust_alarm(rust_task *receiver) :
    receiver(receiver)
{
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
