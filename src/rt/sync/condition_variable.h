#ifndef CONDITION_VARIABLE_H
#define CONDITION_VARIABLE_H

class condition_variable {
#if defined(__WIN32__)
    HANDLE _event;
#else
    pthread_cond_t _cond;
    pthread_mutex_t _mutex;
#endif
public:
    condition_variable();
    virtual ~condition_variable();

    void wait();
    void signal();
};

#endif /* CONDITION_VARIABLE_H */
