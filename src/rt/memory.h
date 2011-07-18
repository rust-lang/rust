// -*- c++ -*-
#ifndef MEMORY_H
#define MEMORY_H

// FIXME: It would be really nice to be able to get rid of this.
inline void *operator new[](size_t size, rust_task *task, const char *tag) {
    return task->malloc(size, tag);
}

template <typename T>
inline void *task_owned<T>::operator new(size_t size, rust_task *task,
                                         const char *tag) {
    return task->malloc(size, tag);
}

template <typename T>
inline void *task_owned<T>::operator new[](size_t size, rust_task *task,
                                           const char *tag) {
    return task->malloc(size, tag);
}

template <typename T>
inline void *task_owned<T>::operator new(size_t size, rust_task &task,
                                         const char *tag) {
    return task.malloc(size, tag);
}

template <typename T>
inline void *task_owned<T>::operator new[](size_t size, rust_task &task,
                                           const char *tag) {
    return task.malloc(size, tag);
}

template <typename T>
inline void *kernel_owned<T>::operator new(size_t size, rust_kernel *kernel,
                                           const char *tag) {
    return kernel->malloc(size, tag);
}


#endif /* MEMORY_H */
