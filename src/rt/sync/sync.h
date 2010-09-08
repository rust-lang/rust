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

/**
 * Thread utility class. Derive and implement your own run() method.
 */
class rust_thread {
public:
#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_t thread;
#endif
    void start();

    virtual void run() {
        return;
    }

    void join();

    bool is_running();
};

#endif /* SYNC_H */
