#include "../globals.h"
#include "sync.h"

void sync::yield() {
#ifdef __APPLE__
    pthread_yield_np();
#elif __WIN32__
    Sleep(1);
#else
    pthread_yield();
#endif
}

void sync::sleep(size_t timeout_in_ms) {
#ifdef __WIN32__
    Sleep(timeout_in_ms);
#else
    usleep(timeout_in_ms * 1000);
#endif
}
