#include "../globals.h"
#include "sync.h"

void sync::yield() {
#ifdef __APPLE__
    pthread_yield_np();
#elif __WIN32__

#else
    pthread_yield();
#endif
}
