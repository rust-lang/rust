fn emulate_foreign_item() {
    match link_name {
        // A comment here will duplicate the attribute
        #[rustfmt::skip]
        | "pthread_mutexattr_init"
        | "pthread_mutexattr_settype"
        | "pthread_mutex_init"
        => {}
    }
}
