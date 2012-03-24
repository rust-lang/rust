#include "rust_internal.h"
#include "rust_shape.h"

void
annihilate_boxes(rust_task *task) {
    LOG(task, gc, "annihilating boxes for task %p", task);

    boxed_region *boxed = &task->boxed;
    rust_opaque_box *box = boxed->first_live_alloc();
    while (box != NULL) {
        rust_opaque_box *tmp = box;
        box = box->next;
        boxed->free(tmp);
    }
}
