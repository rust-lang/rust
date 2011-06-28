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
    return task->malloc(size, memory_region::LOCAL);
}

inline void *operator new[](size_t size, rust_task *task) {
    return task->malloc(size, memory_region::LOCAL);
}

inline void *operator new(size_t size, rust_task &task) {
    return task.malloc(size, memory_region::LOCAL);
}

inline void *operator new[](size_t size, rust_task &task) {
    return task.malloc(size, memory_region::LOCAL);
}

inline void *operator new(size_t size, rust_task *task,
    memory_region::memory_region_type type) {
    return task->malloc(size, type);
}

inline void *operator new[](size_t size, rust_task *task,
    memory_region::memory_region_type type) {
    return task->malloc(size, type);
}

inline void *operator new(size_t size, rust_task &task,
    memory_region::memory_region_type type) {
    return task.malloc(size, type);
}

inline void *operator new[](size_t size, rust_task &task,
    memory_region::memory_region_type type) {
    return task.malloc(size, type);
}

inline void operator delete(void *mem, rust_task *task) {
    task->free(mem, memory_region::LOCAL);
    return;
}

inline void operator delete(void *mem, rust_task *task,
    memory_region::memory_region_type type) {
    task->free(mem, type);
    return;
}

#endif /* MEMORY_H */
