#ifndef RUST_THREAD_H
#define RUST_THREAD_H

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
    rust_thread();
    void start();

    virtual void run() {
        return;
    }

    void join();

    virtual ~rust_thread() {}   // quiet the compiler
};

#endif /* RUST_THREAD_H */
