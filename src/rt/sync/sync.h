#ifndef SYNC_H
#define SYNC_H

class sync {
public:
    static void yield();
    static void sleep(size_t timeout_in_ms);
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
private:
    volatile bool _is_running;
public:
#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_t thread;
#endif
    rust_thread();
    void start();

    virtual void run() {
        return;
    }

    void join();

    bool is_running();

    virtual ~rust_thread() {}   // quiet the compiler
};

#endif /* SYNC_H */
