// -*- c++ -*-
#ifndef MEMORY_H
#define MEMORY_H

inline void *operator new(size_t size, void *mem) {
    return mem;
}

inline void *operator new(size_t size, rust_kernel *kernel) {
    return kernel->malloc(size);
}

inline void *operator new(size_t size, rust_task *task) {
    return task->malloc(size);
}

inline void *operator new[](size_t size, rust_task *task) {
    return task->malloc(size);
}

inline void *operator new(size_t size, rust_task &task) {
    return task.malloc(size);
}

inline void *operator new[](size_t size, rust_task &task) {
    return task.malloc(size);
}

inline void operator delete(void *mem, rust_task *task) {
    task->free(mem);
    return;
}

#endif /* MEMORY_H */
