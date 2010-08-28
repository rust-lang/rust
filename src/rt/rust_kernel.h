#ifndef RUST_KERNEL_H
#define RUST_KERNEL_H

/**
 * A global object shared by all domains.
 */
class rust_kernel {
    memory_region &_region;
    rust_log _log;
public:
    synchronized_indexed_list<rust_dom> domains;
    synchronized_indexed_list<lock_free_queue<rust_message*> > message_queues;
    rust_kernel(rust_srv *srv);
    void register_domain(rust_dom *dom);
    void deregister_domain(rust_dom *dom);
    void log_all_domain_state();
    void log(uint32_t type_bits, char const *fmt, ...);
    virtual ~rust_kernel();
};

#endif /* RUST_KERNEL_H */
