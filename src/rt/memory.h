// -*- c++ -*-
#ifndef MEMORY_H
#define MEMORY_H

// FIXME: It would be really nice to be able to get rid of this.
inline void *operator new[](size_t size, rust_task *task) {
    return task->malloc(size);
}

template <typename T>
inline void *task_owned<T>::operator new(size_t size, rust_task *task) {
    return task->malloc(size);
}

template <typename T>
inline void *task_owned<T>::operator new[](size_t size, rust_task *task) {
    return task->malloc(size);
}

template <typename T>
inline void *task_owned<T>::operator new(size_t size, rust_task &task) {
    return task.malloc(size);
}

template <typename T>
inline void *task_owned<T>::operator new[](size_t size, rust_task &task) {
    return task.malloc(size);
}

template <typename T>
inline void *kernel_owned<T>::operator new(size_t size, rust_kernel *kernel) {
    return kernel->malloc(size);
}


#endif /* MEMORY_H */
