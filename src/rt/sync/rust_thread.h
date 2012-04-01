#ifndef RUST_THREAD_H
#define RUST_THREAD_H

/**
 * Thread utility class. Derive and implement your own run() method.
 */
class rust_thread {
 private:
#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_t thread;
#endif
    size_t stack_sz;
 public:

    rust_thread();
    rust_thread(size_t stack_sz);
    virtual ~rust_thread();

    void start();

    virtual void run() = 0;

    void join();
    void detach();
};

#endif /* RUST_THREAD_H */
