#ifndef SYNC_H
#define SYNC_H

class sync {
public:
    static void yield();
    template <class T>
    static bool compare_and_swap(T *address,
        T oldValue, T newValue) {
        return __sync_bool_compare_and_swap(address, oldValue, newValue);
    }
};

#endif /* SYNC_H */
