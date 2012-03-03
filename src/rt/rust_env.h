struct rust_env {
    size_t num_sched_threads;
    size_t min_stack_size;
    size_t max_stack_size;
    char* logspec;
    bool check_claims;
    bool detailed_leaks;
    char* rust_seed;
    bool poison_on_free;
};

rust_env* load_env();
void free_env(rust_env *rust_env);
