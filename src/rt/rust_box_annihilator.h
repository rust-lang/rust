#ifndef RUST_BOX_ANNIHILATOR_H
#define RUST_BOX_ANNIHILATOR_H

#include "rust_task.h"

void
annihilate_box(rust_task *task, rust_opaque_box *box);

void
annihilate_boxes(rust_task *task);

#endif
