// See comments in build.rs for why this exists
int pthread_atfork(void* prefork,
                   void* postfork_parent,
                   void* postfork_child) {
  return 0;
}
