#include "rust_internal.h"

rust_kernel::rust_kernel(rust_srv *srv) :
    _region(srv->local_region),
    _log(srv, NULL),
    domains(srv->local_region),
    message_queues(srv->local_region) {
    // Nop.
}

rust_kernel::~rust_kernel() {
    // Nop.
}

void
rust_kernel::register_domain(rust_dom *dom) {
    domains.append(dom);
}

void
rust_kernel::deregister_domain(rust_dom *dom) {
    domains.remove(dom);
}

void
rust_kernel::log_all_domain_state() {
    log(rust_log::KERN, "log_all_domain_state: %d domains", domains.length());
    for (uint32_t i = 0; i < domains.length(); i++) {
        domains[i]->log_state();
    }
}

void
rust_kernel::log(uint32_t type_bits, char const *fmt, ...) {
    char buf[256];
    if (_log.is_tracing(type_bits)) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        _log.trace_ln(NULL, type_bits, buf);
        va_end(args);
    }
}
